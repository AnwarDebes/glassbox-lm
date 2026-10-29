"""Tests of reg_table.py on synthetic report files (standard library only).

run: cd studies/agent_vs_llm && ~/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12 -I \
         test_reg_table.py
"""
import contextlib
import io
import json
import os
import sys
import tempfile

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
import reg_table as T  # noqa: E402

OUT = []


def check(name, ok):
    OUT.append(f"{'ok  ' if ok else 'FAIL'} {name}")
    print(OUT[-1])


def rep(ds, diff, lo, hi, tl_mean, tl_lo, tl_hi, ft_lo=0.02):
    return {"dataset": ds, "n_rows": 4000, "acc_agent": 0.70, "acc_llm": 0.70 - diff, "acc_ft": 0.78,
            "agent_minus_llm": diff, "ci_agent_minus_llm": [lo, hi], "ci90_agent_minus_llm": [lo + 0.003, hi - 0.003],
            "NI_met": lo > -0.03, "NI2_met": lo > -0.02, "SUP_met": diff >= 0.03 and lo > 0, "FT_met": ft_lo > 0,
            "EQ_inside_no_rule": lo + 0.003 > -0.03 and hi - 0.003 < 0.03, "ft_minus_agent": 0.08,
            "ci_ft_minus_agent": [ft_lo, 0.1], "llm_median_s_per_row": 11.0, "llm_invalid": 0,
            "tfidf_no_rule": {"acc_tfidf": 0.7, "tfidf_minus_llm": 0.0, "ci_tfidf_minus_llm": [-0.01, 0.01],
                              "agent_minus_tfidf": 0.0, "ci_agent_minus_tfidf": [-0.01, 0.01]},
            "two_level": {"agent_minus_llm_mean": tl_mean, "ci": [tl_lo, tl_hi], "ft_minus_agent_ci": [ft_lo, 0.1],
                          "agent_minus_tfidf_ci": [-0.01, 0.01]}}


with tempfile.TemporaryDirectory() as d:
    json.dump(rep("star", 0.035, 0.005, 0.065, 0.031, 0.001, 0.062), open(os.path.join(d, "report_star.json"), "w"))
    json.dump(rep("abcd", -0.005, -0.025, 0.015, -0.007, -0.031, 0.016), open(os.path.join(d, "report_abcd.json"), "w"))
    json.dump(rep("multiwoz", 0.0, -0.018, 0.018, -0.002, -0.021, 0.017), open(os.path.join(d, "report_multiwoz.json"), "w"))
    with contextlib.redirect_stdout(io.StringIO()):
        rules, norule, claims, notes = T.table(d)
    lines = rules.splitlines()
    check("one row per dataset under the header", len(lines) == 5)
    check("STAR: SUP met on seed 1 and on the 5-seed mean", "| STAR |" in lines[2] and lines[2].count("| met |") >= 3)
    check("ABCD: NI met on seed 1 but not on the 5-seed two-level interval -> marked seed 1 only",
          "met (seed 1 only)" in lines[3] and "ABCD NI: met on seed 1, not on the 5-seed mean" in notes)
    check("MultiWOZ: NI2 met on seed 1, the two-level lower end -0.021 fails it -> seed 1 only",
          "MultiWOZ NI2: met on seed 1, not on the 5-seed mean" in notes)
    check("NI on all three, NI2 not on ABCD (-0.025) -> the general claim at the 3-point margin, NI2 named per dataset",
          any(c.startswith("NI met on all three datasets") and "3 points (NI)" in c for c in claims)
          and "NI2 (2 points) met on: STAR, MultiWOZ; not on: ABCD." in claims)
    json.dump(rep("abcd", -0.005, -0.015, 0.015, -0.005, -0.016, 0.016), open(os.path.join(d, "report_abcd.json"), "w"))
    with contextlib.redirect_stdout(io.StringIO()):
        _, _, claims_b, _ = T.table(d)
    check("NI2 on all three -> the general claim at the 2-point margin",
          any(c.startswith("NI met on all three datasets") and "2 points (NI2)" in c for c in claims_b))
    json.dump(rep("abcd", -0.005, -0.025, 0.015, -0.007, -0.031, 0.016), open(os.path.join(d, "report_abcd.json"), "w"))
    check("SUP claim line for STAR only", sum("SUP met" in c for c in claims) == 1 and any("STAR: SUP met" in c for c in claims))
    json.dump({"gate": {"star": True, "abcd": False, "multiwoz": True}}, open(os.path.join(d, "checks.json"), "w"))
    with contextlib.redirect_stdout(io.StringIO()):
        rules2, _, claims2, _ = T.table(d)
    check("a closed gate shows the dataset as not completed and drops it from the claim",
          "| ABCD | gate closed: not completed" in rules2 and not any(c.startswith("NI met on all three") for c in claims2))
    json.dump({"verdict": "MATCH"}, open(os.path.join(d, "verify_star.out"), "w"))
    with contextlib.redirect_stdout(io.StringIO()):
        rules3, _, _, _ = T.table(d)
    check("the verify verdict is read from verify_<ds>.out", rules3.splitlines()[2].rstrip().endswith("| MATCH |"))
    outp = os.path.join(d, "table.md")
    with contextlib.redirect_stdout(io.StringIO()):
        T.main(["--dir", d, "--out", outp])
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            T.main(["--dir", d, "--out", outp])
        check("an existing table is never overwritten", False)
    except FileExistsError:
        check("an existing table is never overwritten", True)
n_fail = sum(x.startswith("FAIL") for x in OUT)
print(f"{len(OUT) - n_fail} passed, {n_fail} failed")
print("PASSED" if n_fail == 0 else "FAILED")
sys.exit(1 if n_fail else 0)
