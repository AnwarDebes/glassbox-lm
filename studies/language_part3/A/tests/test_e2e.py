"""The dry run: every A-Struct entry point, in the chain's order, on the synthetic corpus (no real data), with tiny
TM settings for the pilot. Checks each stage's outputs, the S2a guard, the refusals of real runs, and the K3c decision
code path (amendment 1: K8 on two equal-size rebuilds; K3c needs a pilot better than B; registered pilot settings)."""

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
from helpers import TEST_TMP, synth_dirs

A = Path(__file__).resolve().parents[1]
PY_TM, PY_DL = os.environ["PY_TM"], os.environ["PY_DL"]


def run(py, script, *args, ok=(0,)):
    t = time.time()
    r = subprocess.run([py, "-I", "-B", str(A / script), *map(str, args)], capture_output=True, text=True, check=False)
    if r.returncode not in ok:
        raise AssertionError(f"{script} {args} exit {r.returncode}\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    print(f"{script} {' '.join(map(str, args))[:80]}: exit {r.returncode}, {time.time() - t:.1f} s")
    return r


@pytest.fixture(scope="module")
def root():
    base = TEST_TMP / "e2e"
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True)
    return base


def test_dry_run(root):
    mw, c4 = synth_dirs("e2e_data", n=200, seed=11)
    R = root / "out"
    R.mkdir()
    run(PY_TM, "s0_data.py", "--mwoz", mw, "--c4", c4, "--out", R / "s0")
    s0 = json.loads((R / "s0" / "done.json").read_text())
    assert s0["count_derived_bits"] == 44 and s0["bits"] - s0["bits_A_nf"] == 44
    fake_log = root / "LOG.md"
    fake_log.write_text("# no hash here\n")
    r = run(PY_TM, "s1_floors.py", "--s0", R / "s0", "--out", R / "s1", "--logmd", fake_log, "--kenlm-discount-fallback", ok=(1,))
    assert "bit-list SHA-256" in r.stderr + r.stdout and not (R / "s1").exists()
    # the chain's helper writes the bit-list line (time from date -u in the same command)
    r = subprocess.run(["bash", str(A / "log_bitlist.sh"), str(fake_log)], env=dict(os.environ, A_S0_DIR=str(R / "s0")),
                       capture_output=True, text=True, check=False)
    assert r.returncode == 0 and s0["bit_list_sha256"] in fake_log.read_text()
    # the dry-run flag is refused for a real output folder (before anything is read or written)
    r = run(PY_TM, "s1_floors.py", "--s0", R / "s0", "--logmd", fake_log, "--kenlm-discount-fallback", ok=(1,))
    assert "dry run only" in r.stderr + r.stdout
    run(PY_TM, "s1_floors.py", "--s0", R / "s0", "--out", R / "s1", "--logmd", fake_log, "--kenlm-discount-fallback")
    s1 = json.loads((R / "s1" / "results.json").read_text())
    assert len(s1["tune_cal_bits"]) == 11 and s1["floor"] in s1["tune_cal_bits"] and s1["K1"]["pass"]
    assert set(s1["picks"]) == {"F2", "F3", "F4"}
    assert len(s1["K2"]) == 22 and all(v["pass"] for v in s1["K2"].values())  # 11 vectors x tune-cal and tune-hold
    # the S2a guard: without the bit-list hash in LOG.md it refuses to start
    fake_log.write_text("# no hash here\n")
    r = run(PY_TM, "s2a_rows.py", "--s0", R / "s0", "--s1", R / "s1", "--out", R / "s2a", "--logmd", fake_log, ok=(1,))
    assert "bit-list SHA-256" in r.stderr + r.stdout and not (R / "s2a").exists()
    fake_log.write_text(f"- test: bit list {s0['bit_list_sha256']}\n")
    run(PY_TM, "s2a_rows.py", "--s0", R / "s0", "--s1", R / "s1", "--out", R / "s2a", "--logmd", fake_log)
    s2a = json.loads((R / "s2a" / "results.json").read_text())
    assert s2a["leak_test"]["pass"] and s2a["nofuture"]["pass"] and s2a["nofuture"]["positions"] > 0
    lk = s2a["leak_test"]
    assert lk["S_lookups"] == lk["S_lookups_unseen"] and lk["S_positions"] == lk["S_positions_count_zero"]
    run(PY_TM, "s2b_twin.py", "--s1", R / "s1", "--s2a", R / "s2a", "--out", R / "s2b_twin", ok=(0, 3))
    tw = json.loads((R / "s2b_twin" / "results.json").read_text())
    assert len(tw["configs"]) == 8 and "K9" in tw
    arm_args = ("--s0", R / "s0", "--s1", R / "s1", "--s2a", R / "s2a", "--twin", R / "s2b_twin", "--aa", R / "s2c_aa")
    # a learner refuses to start before K8 is decided
    r = run(PY_TM, "s2b_arm.py", "--arm", "full", "--family", "lr1", *arm_args, "--out", R / "s2b_full_lr1", ok=(1,))
    assert "S2c (A/A, K8) is not done" in r.stderr + r.stdout and not (R / "s2b_full_lr1.partial").exists()
    run(PY_TM, "s2c_aa.py", "--s0", R / "s0", "--s1", R / "s1", "--twin", R / "s2b_twin", "--out", R / "s2c_aa", ok=(0, 3))
    aa = json.loads((R / "s2c_aa" / "results.json").read_text())
    k8 = json.loads((R / "s2c_aa" / "k8.json").read_text())
    assert aa["K8"] == k8["K8"] and "reported_no_rule" not in k8  # the decision is written before the reported numbers
    assert aa["K8_rule"].startswith("rebuilt(seed 2) minus rebuilt(seed 3)") and aa["seeds"] == [2, 3]
    # amendment 1: the registered K8 is the equal-size A/A, recomputed here from the saved per-position bits
    z = np.load(R / "s2c_aa" / "bits.npz")
    assert abs(aa["K8"]["mean"] - float((z["B_rebuilt_hold"] - z["B_rebuilt3_hold"]).mean())) < 1e-12
    assert not np.array_equal(z["B_rebuilt_hold"], z["B_rebuilt3_hold"])  # two different rebuilds
    ok8 = abs(aa["K8"]["mean"]) <= 0.0036 and aa["K8"]["lo"] <= 0 <= aa["K8"]["hi"]
    assert aa["K8"]["pass"] == ok8
    rp = aa["reported_no_rule"]
    B_hold = np.load(R / "s2b_twin" / "bits.npz")["B_hold"]
    assert abs(rp["B_minus_rebuilt2"]["mean"] - float((B_hold - z["B_rebuilt_hold"]).mean())) < 1e-12
    assert abs(rp["B_minus_rebuilt3"]["mean"] - float((B_hold - z["B_rebuilt3_hold"]).mean())) < 1e-12
    assert aa["dropped_dialogues"]["seed2"] == aa["dropped_dialogues"]["seed3"] > 0
    if not aa["K8"]["pass"]:  # the registered stop holds; the dry run goes on only with its own flag
        r = run(PY_TM, "s2b_arm.py", "--arm", "full", "--family", "lr1", *arm_args, "--out", R / "s2b_full_lr1", ok=(1,))
        assert "K8 STOP" in r.stderr + r.stdout
    r = run(PY_TM, "s2b_arm.py", "--arm", "full", "--family", "lr1", *arm_args, "--dry-run-ignore-k8", ok=(1,))
    assert "dry run only" in r.stderr + r.stdout  # the flag is refused for a real output folder
    for arm in ("full", "nf"):
        for fam in ("tree", "lr1", "lr2", "hgb"):
            run(PY_TM, "s2b_arm.py", "--arm", arm, "--family", fam, *arm_args, "--dry-run-ignore-k8",
                "--out", R / f"s2b_{arm}_{fam}", ok=(0, 3))
            res = json.loads((R / f"s2b_{arm}_{fam}" / "results.json").read_text())
            assert res["nofuture"]["pass"] and res["pick"] in res["configs"]
            assert res["bits_used"] == (s0["bits"] if arm == "full" else s0["bits_A_nf"])
            assert len(res["configs"]) == {"tree": 8, "lr1": 4, "lr2": 4, "hgb": 8}[fam]
    # once a learner has written its outputs, a (second) A/A run is refused
    r = run(PY_TM, "s2c_aa.py", "--s0", R / "s0", "--s1", R / "s1", "--twin", R / "s2b_twin", "--out", R / "s2c_aa2", ok=(1,))
    assert "K8 must be decided before any learner runs" in r.stderr + r.stdout and not (R / "s2c_aa2.partial").exists()
    run(PY_TM, "s2d_gates.py", "--root", R, ok=(0, 3))
    g = json.loads((R / "s2d_gates" / "gates.json").read_text())
    assert set(g["arms"]) == {"full", "nf"} and "K8" in g and "go_to_K3c" in g
    words = (R / "s2d_gates" / "words.txt").read_text()
    assert "K3 (i)" in words and "K8 (A/A, equal size): rebuilt(2) minus rebuilt(3)" in words
    assert "reported, no rule: B minus rebuilt(2)" in words and "reported, no rule: B minus rebuilt(3)" in words
    assert g["K8"] == aa["K8"] and "K8_reported_no_rule" in g


def _crafted_pilots(src: Path, dst: Path, minus_B: dict) -> None:
    """Copies of the two dry pilots whose tune-hold result is set by hand, only to run every branch of the decision."""
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for s, v in minus_B.items():
        shutil.copytree(src / f"k3c_full_s{s}", dst / f"k3c_full_s{s}")
        r = json.loads((dst / f"k3c_full_s{s}" / "results.json").read_text())
        r["tune_hold_minus_B"] = v
        (dst / f"k3c_full_s{s}" / "results.json").write_text(json.dumps(r))


def test_k3c_pilot_and_gate(root):
    import k3c_gate
    import k3c_pilot
    from astruct import paths
    R = root / "out"
    if not (R / "s2d_gates").exists():
        pytest.skip("needs the dry run")
    inputs = ("--s0", R / "s0", "--s1", R / "s1", "--s2a", R / "s2a", "--twin", R / "s2b_twin")
    # amendment 1: a real pilot (default folder under A/out/) with the dry-run settings is refused before it reads or
    # writes anything; so is one before S2d
    r = run(PY_DL, "k3c_pilot.py", "--arm", "full", "--s", 15, "--epochs", 1, "--clauses", 40, *inputs, ok=(1,))
    msg = r.stderr + r.stdout
    # before the real S2d the refusal names it; since the real S2d ran, its gates (go_to_K3c false) are the reason
    assert "real pilot refused" in msg and "--epochs 1 / --clauses 40" in msg and (
        "S2d is not done" in msg or "stopped at the S2 gates (go_to_K3c false)" in msg)
    assert not any(paths.OUT.glob("k3c_*"))
    for s in (15, 40):
        run(PY_DL, "k3c_pilot.py", "--arm", "full", "--s", s, "--epochs", 1, "--clauses", 40, *inputs,
            "--out", R / f"k3c_full_s{s}")
    pil = {s: json.loads((R / f"k3c_full_s{s}" / "results.json").read_text()) for s in (15, 40)}
    best = min((15, 40), key=lambda s: (pil[s]["tune_cal_bits"], s))
    assert pil[best]["clauses"] == 40 and len(pil[best]["epoch_seconds"]) == 1
    assert pil[best]["K2"]["cal"]["pass"] and pil[best]["K2"]["hold"]["pass"] and pil[best]["nofuture"]["pass"]
    # the registered-settings check of the gate: the dry pilots fail it, a registered one passes
    assert k3c_gate.pilot_problems(pil[15], "full", 15, False)
    reg = {**pil[15], "clauses": 4000, "epochs": 10}
    assert k3c_gate.pilot_problems(reg, "full", 15, False) == [] and k3c_gate.pilot_problems(reg, "nf", 15, False)
    assert k3c_gate.pilot_problems(reg, "full", 40, False) and k3c_gate.pilot_problems(reg, "full", 15, True)
    g = json.loads((R / "s2d_gates" / "gates.json").read_text())
    g["go_to_K3c"]["full"] = True  # crafted copies, only to run every branch of the decision code
    # the real-pilot checks on a crafted root: S2d stop, settings, the shuffled pilot's s and K3c
    croot = root / "crafted_real"
    _crafted_pilots(R, croot, {15: -0.5, 40: -0.4})
    (croot / "s2d_gates").mkdir()
    g_stop = json.loads(json.dumps(g))
    g_stop["go_to_K3c"]["full"] = False
    (croot / "s2d_gates" / "gates.json").write_text(json.dumps(g_stop))
    assert any("stopped at the S2 gates" in x for x in
               k3c_pilot.real_pilot_problems("full", 15, False, 10, 4000, croot / "k3c_full_s15", root=croot))
    g_tr = json.loads(json.dumps(g))
    g_tr["arms"]["full"]["families"]["hgb"]["minus_B"]["mean"] = -0.06
    (croot / "s2d_gates" / "gates.json").write_text(json.dumps(g_tr))
    assert k3c_pilot.real_pilot_problems("full", 15, False, 10, 4000, croot / "k3c_full_s15", root=croot) == []
    assert k3c_pilot.real_pilot_problems("full", 15, False, 10, 4000, croot / "elsewhere", root=croot)
    assert k3c_pilot.real_pilot_problems("full", 5, False, 10, 4000, croot / "k3c_full_s5", root=croot)
    c_best = min((15, 40), key=lambda s: (pil[s]["tune_cal_bits"], s))  # the copies keep the tune-cal bits
    other = 40 if c_best == 15 else 15
    assert k3c_pilot.real_pilot_problems("full", c_best, True, 10, 4000, croot / f"k3c_full_s{c_best}_shuffled",
                                         root=croot) == []
    assert any("better s" in x for x in k3c_pilot.real_pilot_problems(
        "full", other, True, 10, 4000, croot / f"k3c_full_s{other}_shuffled", root=croot))
    g_tr["arms"]["full"]["families"]["hgb"]["minus_B"]["mean"] = -10.0  # no pilot reaches a third of 10 bits
    (croot / "s2d_gates" / "gates.json").write_text(json.dumps(g_tr))
    assert any("K3c stopped" in x for x in k3c_pilot.real_pilot_problems(
        "full", c_best, True, 10, 4000, croot / f"k3c_full_s{c_best}_shuffled", root=croot))
    # a real K3c decision reads only A/out/: refused here before anything is written
    r = run(PY_TM, "k3c_gate.py", "--arm", "full", "--root", R, "--out", paths.OUT / "k3c_gate_full", ok=(1,))
    assert "reads only A/out/" in r.stderr + r.stdout and not any(paths.OUT.glob("k3c_gate_*"))
    # branch 1: the trees gained 10 bits over B, so no pilot reaches a third of that: K3c STOP, no shuffled pilot needed
    g["arms"]["full"]["families"]["hgb"]["minus_B"]["mean"] = -10.0
    (root / "gates_stop.json").write_text(json.dumps(g))
    run(PY_TM, "k3c_gate.py", "--arm", "full", "--root", R, "--gates", root / "gates_stop.json", "--out", root / "gate_stop", ok=(3,))
    k = json.loads((root / "gate_stop" / "k3c.json").read_text())
    assert not k["K3c"]["pass"] and k["K9_tm"] is None and not k["go_to_GPU"] and k["better_s"] == best
    # branch 2 (amendment 1): the trees lost 10 bits and the pilot is worse than B: K3c STOP ("a third of a negative
    # gain" no longer passes)
    g["arms"]["full"]["families"]["hgb"]["minus_B"]["mean"] = 10.0
    (root / "gates_pass.json").write_text(json.dumps(g))
    worse = root / "crafted_worse"
    _crafted_pilots(R, worse, {15: 0.010, 40: 0.020})
    run(PY_TM, "k3c_gate.py", "--arm", "full", "--root", worse, "--gates", root / "gates_pass.json", "--out",
        root / "gate_worse", ok=(3,))
    k = json.loads((root / "gate_worse" / "k3c.json").read_text())
    assert not k["K3c"]["pass"] and not k["K3c"]["pilot_beats_B"] and k["K3c"]["third_of_trees_gain"]
    # branch 3: the trees lost 10 bits and the pilot beats B: K3c passes; first the gate asks for the shuffled pilot
    # and writes nothing, then it decides with the shuffled pilot's K9
    better = root / "crafted_better"
    _crafted_pilots(R, better, {15: -0.5, 40: -0.4})
    r = run(PY_TM, "k3c_gate.py", "--arm", "full", "--root", better, "--gates", root / "gates_pass.json", "--out",
            root / "gate_pass", ok=(5,))
    assert f"--s {best} --shuffled" in r.stdout and not (root / "gate_pass").exists()
    run(PY_DL, "k3c_pilot.py", "--arm", "full", "--s", best, "--shuffled", "--epochs", 1, "--clauses", 40, *inputs,
        "--out", better / f"k3c_full_s{best}_shuffled")
    run(PY_TM, "k3c_gate.py", "--arm", "full", "--root", better, "--gates", root / "gates_pass.json", "--out",
        root / "gate_pass", ok=(0, 3))
    k = json.loads((root / "gate_pass" / "k3c.json").read_text())
    assert k["K3c"]["pass"] and k["K3c"]["pilot_beats_B"] and k["K9_tm"] is not None
    assert k["go_to_GPU"] == k["K9_tm"]["pass"] and k["engine"].startswith("our CoTM")
