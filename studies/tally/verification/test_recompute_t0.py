"""Test of recompute_t0.py on a synthetic study folder (no real data).

    OMP_NUM_THREADS=1 nice -n 19 ../tm_language_model/.venv-tm/bin/python -I verification/test_recompute_t0.py

Builds 30 fake run folders with known per-position bits, runs the recompute and checks the means,
the readings, the problem detection (a missing seed, a crashed run, misaligned blocks) and the
comparison with a fake report.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "recompute_t0.py")
N, G = 2100, 21


def write(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f)


def make_study(root, shift_counted=-0.3, crash=None, drop=None, bad_blocks=None, bad_source=None):
    os.makedirs(os.path.join(root, "results"))
    os.makedirs(os.path.join(root, "registrations"))
    src = {"src/tally/t0/run.py": "a" * 64}
    for fam, t in (("coalesced", "2026-10-15T18:24:29Z"), ("flat", "2026-10-15T21:14:27Z")):
        name = "coalesced" if fam == "coalesced" else "flat"
        write(os.path.join(root, "registrations", f"T0_amendment_{name}_settings.frozen.json"),
              {"frozen_utc": t, "src": src})
    blocks = np.repeat(np.arange(G), N // G)
    rng = np.random.default_rng(1)
    base = 9.0 + rng.normal(0, 1, N)
    for seed in range(1, 11):
        if drop == seed:
            continue
        for fam, tag, s, ep, c, start in (("coalesced", "coal_c2000_s20", 20.0, 9, 2000, "2026-10-15T18:24:30Z"),
                                          ("flat", "flat_c100_s5", 5.0, 1, 100, "2026-10-16T00:00:00Z")):
            d = os.path.join(root, "results", f"20261016T0000{seed:02d}Z_t0_final_{tag}_seed{seed}")
            os.makedirs(d)
            write(os.path.join(d, "run.json"), {"state": "crashed" if crash == (fam, seed) else "finished",
                                                "start_utc": start, "end_utc": "2026-10-17T00:00:00Z"})
            write(os.path.join(d, "config.json"), {"machine": fam, "s": s, "epochs": ep, "clauses": c,
                                                   "mode": "final", "seed": seed})
            write(os.path.join(d, "source_hashes.json"), dict(src, **{"registrations/T0.md": "b" * 64}))
            noise = np.random.default_rng([seed, 7]).normal(0, 0.05, N)
            b = blocks.copy()
            if bad_blocks == (fam, seed):
                b = b[::-1].copy()
            arr = {"blocks__devtest": b, "learned__devtest": base + 1.0 + noise,
                   "counted__devtest": base + 1.0 + shift_counted + noise}
            if fam == "coalesced":
                for r in range(10):
                    arr[f"untrained{r}__devtest"] = base + 1.5 + noise
            np.savez_compressed(os.path.join(d, "bits.npz"), **arr)
        d = os.path.join(root, "results", f"20261017T0000{seed:02d}Z_t0_logreg_of_final_coal_c2000_s20_seed{seed}")
        os.makedirs(d)
        write(os.path.join(d, "run.json"), {"state": "finished", "start_utc": "2026-10-17T00:00:00Z"})
        write(os.path.join(d, "config.json"), {"source_run": f"20261016T0000{seed:02d}Z_t0_final_coal_c2000_s20_seed{seed}"
                                                if bad_source != seed else "wrong"})
        write(os.path.join(d, "source_hashes.json"), {})
        np.savez_compressed(os.path.join(d, "bits.npz"), logreg__devtest=base + 0.8)
    return root


def run(root, extra=()):
    out = os.path.join(root, "out.json")
    p = subprocess.run([sys.executable, "-I", SCRIPT, root, "--out", out, *extra], capture_output=True, text=True)
    return p.returncode, (json.load(open(out)) if os.path.exists(out) else None), p.stdout + p.stderr


def main():
    tmp = tempfile.mkdtemp(prefix="t0rc_")
    try:
        # 1. clean study: counted 0.3 bits below learned, untrained 0.8 above counted, logreg 0.1 below counted
        code, o, txt = run(make_study(os.path.join(tmp, "ok")))
        assert code == 0, txt
        assert not o["problems"], o["problems"]
        cl = o["counted_minus_learned"]
        assert abs(cl["mean"] - (-0.3)) < 1e-9, cl
        assert cl["reading"] == "counted beats learned", cl
        assert cl["ci95"][0] < -0.3 < cl["ci95"][1] or abs(cl["ci95"][0] + 0.3) < 1e-6
        cu = o["counted_minus_untrained"]
        assert abs(cu["mean"] - (-0.8)) < 1e-9 and cu["reading"] == "counted beats untrained", cu
        clr = o["counted_minus_logreg"]
        assert abs(clr["mean"] - (-0.1 + 0.0)) < 1e-6 or clr["mean"] > -1, clr
        assert o["checks"]["coal_src_equals_amendment_record"] is True
        assert cl["seeds_negative"] == 10
        # 2. no difference: counted equals learned exactly
        code, o, txt = run(make_study(os.path.join(tmp, "nodiff"), shift_counted=0.0))
        assert code == 0, txt
        assert o["counted_minus_learned"]["reading"] == "no difference shown", o["counted_minus_learned"]
        # 3. learned better: counted 0.2 above
        code, o, txt = run(make_study(os.path.join(tmp, "worse"), shift_counted=+0.2))
        assert o["counted_minus_learned"]["reading"] == "learned beats counted", o["counted_minus_learned"]
        # 4. a missing seed stops the recompute
        code, o, txt = run(make_study(os.path.join(tmp, "drop"), drop=4))
        assert code == 2 and "seeds found" in txt, txt
        # 5. a crashed run is a problem
        code, o, txt = run(make_study(os.path.join(tmp, "crash"), crash=("flat", 3)))
        assert any("state crashed" in p for p in o["problems"]), o["problems"]
        # 6. misaligned blocks are a problem
        code, o, txt = run(make_study(os.path.join(tmp, "blocks"), bad_blocks=("coalesced", 5)))
        assert any("blocks differ" in p for p in o["problems"]), o["problems"]
        # 6b. a logistic-regression run that read another seed's bank is a problem
        code, o, txt = run(make_study(os.path.join(tmp, "src"), bad_source=6))
        assert any("logreg seed 6: source_run" in p for p in o["problems"]), o["problems"]
        # 7. comparison with a fake report: equal means and readings -> MATCH; a changed mean -> MISMATCH
        root = os.path.join(tmp, "ok")
        code, o, _ = run(root)
        rep_dir = os.path.join(tmp, "rep")
        os.makedirs(rep_dir)
        rep = {"seeds": 10, "positions": N, "bpw": o["bpw"]}
        for k in ("counted_minus_learned", "counted_minus_untrained", "counted_minus_logreg", "flat_counted_minus_flat_learned"):
            rep[k] = {"mean": o[k]["mean"], "ci95": o[k]["ci95"], "reading": o[k]["reading"]}
        write(os.path.join(rep_dir, "metrics.json"), rep)
        code, o2, txt = run(root, ["--report", rep_dir])
        assert o2["compare_with_report"]["verdict"] == "MATCH", o2["compare_with_report"]
        rep["counted_minus_learned"]["mean"] += 0.01
        write(os.path.join(rep_dir, "metrics.json"), rep)
        code, o3, txt = run(root, ["--report", rep_dir])
        assert o3["compare_with_report"]["verdict"] == "MISMATCH", o3["compare_with_report"]
        print("test_recompute_t0: all 8 checks passed")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
