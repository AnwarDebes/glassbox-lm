"""Tests of the REG_agent_vs_llm tools: reg_report.py, verify_reg.py, rows_rule.py, make_reg_jobs.py, check_reg.py.

run: cd studies/agent_vs_llm && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 19 \
         ../decision_layer/.venv-dl/bin/python -I test_reg_tools.py
Dev files only: no test split is read (the test-split guards are tested on refusals that happen before any read).
1. rules(): NI at -0.03, NI2 at -0.02, SUP, FT and the equivalence reading at the edges.
2. registered_rows() is the draw llm_next_action.py makes with --seed 20261027 and no exclusion.
3. End to end on ABCD dev (agent seed 1, L35k50, fine-tuned bge-base, TFIDF): reg_report.py's accuracies equal a plain
   recount; the agent in the registered seed-file format (pred + y) gives the same report; the 90% interval lies inside
   the 95% one; the rules follow from the intervals; verify_reg.py says MATCH (both formats) and MISMATCH on a flipped
   rule; on the test split both refuse without a freeze record and with a changed registration.
4. rows_rule.py: the power formula, the choices at the edges (ABCD's split used whole), an end-to-end run.
5. make_reg_jobs.py: 3 FT, 3 TFIDF, 3 LLM lines and the report line; reg_report_all.sh parses and holds the checks,
   the three reports and the three recomputes.
6. check_reg.py on a synthetic run folder: all hard checks pass; an LLM row with a request error, a missing row, or a
   dry-run FT result fails them.
"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
_HOME = _os.path.expanduser("~")
import json
import math
import os
import random
import subprocess
import sys
import tempfile

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
import reg_report as R  # noqa: E402

PY312 = f"{_HOME}/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12"
OUT = []


def check(name, ok):
    OUT.append(f"{'ok  ' if ok else 'FAIL'} {name}")
    print(OUT[-1])
    return ok


def test_rules():
    r = R.rules(0.0, [-0.0299, 0.03], [-0.025, 0.025], [0.01, 0.05])
    check("NI met at a lower end of -0.0299; NI2 not", r["NI_met"] and not r["NI2_met"])
    check("FT met when its lower end is +0.01", r["FT_met"])
    check("SUP not met at a mean of 0", not r["SUP_met"])
    check("equivalence inside when the 90% interval is (-0.025, +0.025)", r["EQ_inside_no_rule"])
    r = R.rules(0.0, [-0.0199, 0.03], [-0.015, 0.02], [0.0, 0.05])
    check("NI2 met at a lower end of -0.0199", r["NI2_met"] and r["NI_met"])
    r = R.rules(0.0, [-0.0301, 0.03], [-0.031, 0.02], [0.0, 0.05])
    check("NI not met when the lower end is -0.0301", not r["NI_met"] and not r["NI2_met"])
    check("FT not met when its lower end is exactly 0", not r["FT_met"])
    check("equivalence not inside when the 90% lower end is -0.031", not r["EQ_inside_no_rule"])
    r = R.rules(0.03, [0.001, 0.06], [0.005, 0.055], [0.01, 0.05])
    check("SUP met at +0.03 with the lower end above 0", r["SUP_met"])
    r = R.rules(0.0299, [0.001, 0.06], [0.005, 0.05], [0.01, 0.05])
    check("SUP not met at +0.0299", not r["SUP_met"])
    import verify_reg as V
    check("verify SUP: agreement is plain agreement", V.sup_agrees(0.04, 0.01, 0.01, True) == (True, None))
    ok, note = V.sup_agrees(0.03 - 1e-15, 0.01, 0.01, True)
    check("verify SUP: a difference of +0.03 up to rounding is NEAR-BOUND, not MISMATCH", ok and "rounding" in note)
    check("verify SUP: a clear disagreement stays a MISMATCH", V.sup_agrees(0.02, 0.01, 0.01, True) == (False, None))
    ok, note = V.sup_agrees(0.04, -0.001, 0.001, True)
    check("verify SUP: a lower end within 0.003 of 0 is NEAR-BOUND", ok and "lower end" in note)


def test_draw():
    src = open(os.path.join(H, "llm_next_action.py")).read()
    check("llm_next_action.py draws rows with random.Random(a.seed).sample(pool, ...)",
          "idx = sorted(random.Random(a.seed).sample(pool, min(a.n, len(pool))))" in src
          and "pool = [i for i in range(len(dv)) if i not in excluded]" in src)
    for n_rows, n in ((6988, 2000), (7355, 4000), (3608, 3608)):
        want = sorted(random.Random(20261027).sample(list(range(n_rows)), min(n, n_rows)))
        check(f"registered_rows({n_rows}, {n}) equals the LLM script's draw", R.registered_rows(n_rows, n) == want)


def test_end_to_end():
    import numpy as np
    agent = os.path.join(H, "runs/agent_abcd_c4_tm_s1/tm.npz")
    llm = os.path.join(H, "runs/llm_q3_L35k50/abcd/preds.jsonl")
    ft = os.path.join(H, "runs/ft_abcd_bgebase_s1/dev_preds.npz")
    tfidf = os.path.join(H, "runs/dry/tfidf_abcd/dev_preds.npz")
    with tempfile.TemporaryDirectory() as tmp:
        A = np.load(agent)
        agent_r2x = os.path.join(tmp, "seed1.npz")  # the registered seed-file format
        np.savez_compressed(agent_r2x, pred=A["top_ids"][:, 0].astype(np.int64), y=A["gold"].astype(np.int64),
                            convo_id=np.zeros(len(A["gold"]), np.int64))
        reps = {}
        for name, af in (("topids", agent), ("r2x", agent_r2x)):
            rep = os.path.join(tmp, f"report_{name}.json")
            p = subprocess.run([sys.executable, "-I", os.path.join(H, "reg_report.py"), "--dataset", "abcd", "--split",
                                "dev", "--agent", af, "--llm", llm, "--ft", ft, "--tfidf", tfidf, "--draws", "2000",
                                "--out", rep, "--agent-seeds", agent, agent_r2x], capture_output=True, text=True)
            if not check(f"reg_report.py runs on ABCD dev ({name} format)", p.returncode == 0):
                print(p.stderr[-2000:])
                return
            reps[name] = json.load(open(rep))
        r = reps["topids"]
        same = {k: r[k] for k in ("acc_agent", "acc_llm", "acc_ft", "ci_agent_minus_llm", "NI_met")} == \
               {k: reps["r2x"][k] for k in ("acc_agent", "acc_llm", "acc_ft", "ci_agent_minus_llm", "NI_met")}
        check("the seed-file format (pred + y) gives the same report", same)
        F = np.load(ft)
        T = np.load(tfidf)
        recs = [json.loads(x) for x in open(llm) if x.strip()]
        n_ok = sum(int(A["top_ids"][x["i"], 0] == A["gold"][x["i"]]) for x in recs)
        f_ok = sum(int(F["top_ids"][x["i"], 0] == F["gold"][x["i"]]) for x in recs)
        l_ok = sum(int(x["pred"] == x["gold"]) for x in recs)
        t_ok = sum(int(T["top_ids"][x["i"], 0] == T["gold"][x["i"]]) for x in recs)
        check(f"recount agent {n_ok} LLM {l_ok} FT {f_ok} TFIDF {t_ok} of 500 equals the report",
              r["n_rows"] == 500 and abs(r["acc_agent"] - n_ok / 500) < 1e-12 and abs(r["acc_llm"] - l_ok / 500) < 1e-12
              and abs(r["acc_ft"] - f_ok / 500) < 1e-12 and abs(r["tfidf_no_rule"]["acc_tfidf"] - t_ok / 500) < 1e-12)
        check("dev values equal the dev table (agent 0.858, L35k50 0.864, bge-base 0.898, TFIDF 0.866)",
              (round(r["acc_agent"], 3), round(r["acc_llm"], 3), round(r["acc_ft"], 3),
               round(r["tfidf_no_rule"]["acc_tfidf"], 3)) == (0.858, 0.864, 0.898, 0.866))
        lo95, hi95 = r["ci_agent_minus_llm"]
        lo90, hi90 = r["ci90_agent_minus_llm"]
        check(f"90% interval ({lo90:+.4f}, {hi90:+.4f}) inside the 95% one ({lo95:+.4f}, {hi95:+.4f})",
              lo95 <= lo90 < hi90 <= hi95)
        check("NI, NI2 and EQ follow the intervals", r["NI_met"] == (lo95 > -0.03) and r["NI2_met"] == (lo95 > -0.02)
              and r["EQ_inside_no_rule"] == (lo90 > -0.03 and hi90 < 0.03))
        tl = r["two_level"]
        check("two-level intervals for AGENT-LLM*, FT-AGENT and AGENT-TFIDF, 2 seeds",
              tl["agent_seeds"] == 2 and len(tl["ci"]) == 2 and len(tl["ft_minus_agent_ci"]) == 2
              and len(tl["agent_minus_tfidf_ci"]) == 2 and tl["agent_acc_per_seed"][0] == tl["agent_acc_per_seed"][1])
        check("rows_are_registered_draw None and no LLM error on dev", r["rows_are_registered_draw"] is None
              and r["llm_errors"] == 0)
        for name, af in (("topids", agent), ("r2x", agent_r2x)):
            v = subprocess.run([PY312, "-I", os.path.join(H, "verify_reg.py"), "--dataset", "abcd", "--split", "dev",
                                "--report", os.path.join(tmp, f"report_{name}.json"), "--agent", af, "--llm", llm,
                                "--ft", ft, "--tfidf", tfidf, "--draws", "2000"], capture_output=True, text=True)
            out = json.loads(v.stdout) if v.stdout.strip().startswith("{") else {}
            check(f"verify_reg.py (stdlib, seed 7) says MATCH ({name} format)", v.returncode == 0
                  and out.get("verdict") == "MATCH")
            if out.get("verdict") != "MATCH":
                print(v.stdout[-3000:], v.stderr[-2000:])
        bad = dict(r)
        bad["NI2_met"] = not r["NI2_met"]
        bad["ci_agent_minus_llm"] = [r["ci_agent_minus_llm"][0] + 0.05, r["ci_agent_minus_llm"][1]]
        rep2 = os.path.join(tmp, "report_bad.json")
        json.dump(bad, open(rep2, "w"))
        v2 = subprocess.run([PY312, "-I", os.path.join(H, "verify_reg.py"), "--dataset", "abcd", "--split", "dev",
                             "--report", rep2, "--agent", agent, "--llm", llm, "--ft", ft, "--draws", "500"],
                            capture_output=True, text=True)
        check("verify_reg.py says MISMATCH on a changed interval and a flipped rule", v2.returncode == 1)
        reg = os.path.join(tmp, "REG_x.md")
        open(reg, "w").write("text\n")
        common = ["--dataset", "abcd", "--split", "test", "--n", "3608", "--reg", reg, "--agent", agent, "--llm", llm,
                  "--ft", ft, "--out", os.path.join(tmp, "t.json")]
        p = subprocess.run([sys.executable, "-I", os.path.join(H, "reg_report.py")] + common, capture_output=True, text=True)
        check("reg_report.py refuses the test split without a freeze record", p.returncode != 0 and "no freeze record" in p.stderr)
        json.dump({"sha256": "0" * 64, "frozen_utc": "x"}, open(os.path.join(tmp, "REG_x.frozen.json"), "w"))
        p = subprocess.run([sys.executable, "-I", os.path.join(H, "reg_report.py")] + common, capture_output=True, text=True)
        check("reg_report.py refuses the test split when the registration differs from its record",
              p.returncode != 0 and "differs from its freeze record" in p.stderr)
        vcommon = [x for x in common if x not in ("--out", os.path.join(tmp, "t.json"))] + ["--report", rep2]
        p = subprocess.run([PY312, "-I", os.path.join(H, "verify_reg.py")] + vcommon, capture_output=True, text=True)
        check("verify_reg.py refuses the test split when the registration differs from its record",
              p.returncode != 0 and "differs from its freeze record" in p.stderr)


def test_rows_rule():
    import rows_rule as RR
    se = 0.361 * 1.1 / math.sqrt(2000)
    want = 0.5 * (1 + math.erf((0.03 / se - 1.96) / math.sqrt(2)))
    check(f"power(0, 0.361, 2000) = {RR.power(0.0, 0.361, 2000):.4f} equals the hand value", abs(RR.power(0.0, 0.361, 2000) - want) < 1e-12)
    n, pw = RR.choose(-0.006, 0.361, 3608)
    check(f"ABCD (split of 3,608 rows): used whole, power {pw:.3f}", n == 3608 and pw > 0.94)
    n, pw = RR.choose(0.028, 0.477, 6988)
    check(f"STAR-like (+2.8, sd 0.477): no credit for the lead, 4000 rows, power {pw:.3f}", n == 4000 and pw >= 0.90
          and RR.power(0.0, 0.477, 3000) < 0.90)
    n, pw = RR.choose(0.026, 0.553, 7355)
    check(f"MultiWOZ-like (+2.6, sd 0.553): capped at 4000, power {pw:.3f} below 0.90", n == 4000 and pw < 0.90)
    n, pw = RR.choose(0.0, 0.3, 6988)
    check("a small sd: 2000 rows", n == 2000)
    choice = {"datasets": {
        "star": {"llm_star": {"config": "L35k50", "path": "runs/llm_q3_L35k50/star"}},
        "abcd": {"llm_star": {"config": "L35k50", "path": "runs/llm_q3_L35k50/abcd"}},
        "multiwoz": {"llm_star": {"config": "L35", "path": "runs/llm_mwoz_dev/L35"}}}}
    with tempfile.TemporaryDirectory() as tmp:
        cp, out = os.path.join(tmp, "choice.json"), os.path.join(tmp, "rows.json")
        json.dump(choice, open(cp, "w"))
        p = subprocess.run([PY312, "-I", os.path.join(H, "rows_rule.py"), "--choice", cp, "--out", out],
                           capture_output=True, text=True)
        if not check("rows_rule.py runs end to end (its agent dev files are the registered dev runs)", p.returncode == 0):
            print(p.stderr[-2000:])
            return
        r = json.load(open(out))["datasets"]
        got = tuple(round(r[d]["dev_agent_minus_llm"], 3) for d in ("star", "abcd", "multiwoz"))
        check(f"dev differences {got} equal the dev table (+0.028, -0.006, +0.026)", got == (0.028, -0.006, 0.026))
        check("rows 4000 / 3608 / 4000 for these configurations",
              (r["star"]["rows"], r["abcd"]["rows"], r["multiwoz"]["rows"]) == (4000, 3608, 4000))


def test_jobs():
    choice = {"datasets": {
        "star": {"llm_star": {"config": "L35k50", "model": "Qwen3.6-35B-A3B-Q4_K_M", "thinking": False, "k": 50,
                              "policy_text": True, "max_tokens": None}},
        "abcd": {"llm_star": {"config": "L35t16", "model": "Qwen3.6-35B-A3B-Q4_K_M", "thinking": True, "k": 10,
                              "policy_text": True, "max_tokens": 16384}},
        "multiwoz": {"llm_star": {"config": "L9-noG", "model": "Qwen3.5-9B-Q8_0", "thinking": False, "k": 10,
                                  "policy_text": False, "max_tokens": None}}}}
    with tempfile.TemporaryDirectory() as tmp:
        cp, out, rp = os.path.join(tmp, "choice.json"), os.path.join(tmp, "jobs.txt"), os.path.join(tmp, "rows.json")
        json.dump(choice, open(cp, "w"))
        json.dump({"datasets": {"star": {"rows": 4000}, "abcd": {"rows": 3608}, "multiwoz": {"rows": 2000}}}, open(rp, "w"))
        p = subprocess.run([PY312, "-I", os.path.join(H, "make_reg_jobs.py"), "--choice", cp, "--rows", rp, "--reg",
                            "REG_agent_vs_llm.md", "--out", out], capture_output=True, text=True, cwd=H)
        if not check("make_reg_jobs.py runs", p.returncode == 0):
            print(p.stderr[-2000:])
            return
        lines = [x for x in open(out).read().splitlines() if x.strip()]
        names = [x.split()[0] for x in lines]
        check("3 FT, 3 TFIDF, 3 LLM lines, then the report line (no agent training)",
              [n for n in names if n.startswith("reg_ft_")] == ["reg_ft_star", "reg_ft_abcd", "reg_ft_multiwoz"]
              and sum(n.startswith("reg_tfidf_") for n in names) == 3 and sum(n.startswith("reg_llm_") for n in names) == 3
              and names[-1] == "reg_report_all" and not any(n.startswith("reg_agent") for n in names) and len(names) == 10)
        llm = {x.split()[0]: x for x in lines if x.startswith("reg_llm_")}
        check("LLM lines: reg_llm_job.sh with the rows 4000 / 3608 / 2000",
              all("reg_llm_job.sh" in x for x in llm.values()) and " star " in llm["reg_llm_star"]
              and " 4000 " in llm["reg_llm_star"] and " 3608 " in llm["reg_llm_abcd"] and " 2000 " in llm["reg_llm_multiwoz"])
        check("STAR: k 50, no --think; ABCD: --think --max-tokens 16384, budget -1; MultiWOZ: 9B, --no-guidelines",
              llm["reg_llm_star"].endswith("--k 50") and "--k 10 --think --max-tokens 16384" in llm["reg_llm_abcd"]
              and " Q4_K_M -1 " in llm["reg_llm_abcd"] and "Qwen3.5-9B-Q8_0.gguf Q8_0 0 " in llm["reg_llm_multiwoz"]
              and "--no-guidelines" in llm["reg_llm_multiwoz"])
        sh = os.path.join(tmp, "reg_report_all.sh")
        body = open(sh).read()
        check("reg_report_all.sh parses", subprocess.run(["bash", "-n", sh]).returncode == 0)
        check("reg_report_all.sh: check_reg first, then 3 reports and 3 recomputes with the seed files",
              body.index("check_reg.py") < body.index("reg_report.py") and body.count("reg_report.py ") == 3
              and body.count("verify_reg.py ") == 3 and body.count("R22star/tm_mem_c4_s16/seed5.npz") == 1
              and "--n 3608" in body)
        check("every FT and TFIDF line has --test-registration and no --dry-run-dev",
              all("--test-registration" in x and "--dry-run-dev" not in x for x in lines if x.startswith(("reg_ft", "reg_tfidf"))))
        rec = os.path.join(H, "REG_agent_vs_llm.frozen.json")
        jl = [x for x in lines if not x.startswith("reg_report_all")]
        check("every job line starts with the pin check of the freeze record (its own output file), then &&",
              all(x.split(" ", 1)[1].startswith("cmd:mkdir -p ") and f"freeze_reg.py --check {rec} > " in x
                  and f"runs/reg/pincheck_{x.split()[0]}.out 2>&1 && " in x
                  and x.index("freeze_reg.py --check") < x.index(("ft_baseline.py", "tfidf_reg.py", "reg_llm_job.sh")[
                      0 if x.startswith("reg_ft") else 1 if x.startswith("reg_tfidf") else 2]) for x in jl))
        check("the registration path is absolute in every line and in the report script",
              all(f"{H}/REG_agent_vs_llm.md" in x for x in jl) and " REG_agent_vs_llm.md" not in "".join(jl)
              and f"--reg {H}/REG_agent_vs_llm.md" in body)
        check("reg_report_all.sh: check_reg with the choice and the pinned job file, then a gate per dataset",
              "--choice " in body and f"--jobs {out}" in body and all(f"if gate {d}; then" in body for d in ("star", "abcd", "multiwoz"))
              and [x.strip() for x in body.splitlines()].count("else") == 3 and "gate closed" in body)
    src = open(os.path.join(H, "reg_llm_job.sh")).read()
    check("reg_llm_job.sh: a second attempt only after exit 4, server stopped after each attempt",
          '[ "${rc}" -eq 4 ] || break' in src and src.count('serve_headline.sh" stop') == 1 and "for attempt in 1 2" in src)


def test_check_reg():
    import numpy as np
    import check_reg as C
    with tempfile.TemporaryDirectory() as tmp:
        reg = os.path.join(tmp, "REG_x.md")
        open(reg, "w").write("text\n")
        json.dump({"sha256": "x", "frozen_utc": "x", "pins": {}, "registration_path": reg}, open(os.path.join(tmp, "REG_x.frozen.json"), "w"))
        rows = {"star": 7, "abcd": 5, "multiwoz": 6}
        rp = os.path.join(tmp, "rows.json")
        json.dump({"datasets": {k: {"rows": v} for k, v in rows.items()}}, open(rp, "w"))
        cfg = {"config": "L35k50", "model": "Qwen3.6-35B-A3B-Q4_K_M", "thinking": False, "k": 50, "policy_text": True,
               "max_tokens": None}
        cp = os.path.join(tmp, "choice.json")
        json.dump({"datasets": {d: {"llm_star": cfg} for d in rows}}, open(cp, "w"))
        jobs = {}
        for d in rows:
            jobs[f"reg_ft_{d}"] = f"cmd:echo ft {d}"
            jobs[f"reg_tfidf_{d}"] = f"cmd:echo tfidf {d}"
            jobs[f"reg_llm_{d}"] = f"cmd:echo llm {d} --k 50"
        jp, qp = os.path.join(tmp, "jobs_reg.txt"), os.path.join(tmp, "queue.log")
        open(jp, "w").write("\n".join(f"{n} {a}" for n, a in jobs.items()) + "\n")
        qlog = "".join(f"2026-10-28T00:00:00Z {n} start: {a} (gate: 2026-10-28T00:00:00Z cores 0.1 over 10.0 s)\n"
                       f"2026-10-28T00:10:00Z {n} exit 0\n" for n, a in jobs.items())
        open(qp, "w").write("2026-10-28T00:00:00Z queue start: jobs\n" + qlog)
        runs = os.path.join(tmp, "runs")
        C.FT_DEV  # the FT flag reference stays as it is
        for ds, n in rows.items():
            want = sorted(random.Random(20261027).sample(range(C.N_TEST[ds]), n))
            d = os.path.join(runs, f"llm_{ds}")
            os.makedirs(d)
            with open(os.path.join(d, "preds.jsonl"), "w") as h:
                h.writelines(json.dumps({"i": i, "pred": "a", "gold": "a", "error": None, "s": 1.0}) + "\n" for i in want)
            json.dump({"n": n, "errors": 0, "k": 50, "think": False,
                       "argv": ["llm_next_action.py", "--dataset", ds, "--n", str(n), "--seed", "20261027", "--k", "50",
                                "--url", "u", "--test-registration", reg, "--out", d]},
                      open(os.path.join(d, "result.json"), "w"))
            for arm in ("ft", "tfidf"):
                q = os.path.join(runs, f"{arm}_{ds}")
                os.makedirs(q)
                np.savez(os.path.join(q, "test_preds.npz"), top_ids=np.zeros((3, 1), np.int16), gold=np.zeros(3, np.int16))
                r = {"argv": ["x.py", "--test-registration", reg], "best_dev_acc": C.FT_DEV[ds], "dev_within_0.5_points": True}
                r.update(C.FT_SETTINGS if arm == "ft" else {"C": C.TFIDF_C[ds], "dry_run_dev": False, "registration": reg})
                json.dump(r, open(os.path.join(q, "result.json"), "w"))
        cj = os.path.join(tmp, "c.json")
        args = ["--reg", reg, "--rows", rp, "--choice", cp, "--jobs", jp, "--queue-log", qp, "--runs", runs, "--out", cj,
                "--skip-pins"]

        def run():
            with __import__("contextlib").redirect_stdout(__import__("io").StringIO()):
                rc = C.main(args)
            return rc, json.load(open(cj))["gate"]

        rc, g = run()
        check("check_reg: all hard checks pass and every gate is open on a complete synthetic run folder",
              rc == 0 and all(g.values()))
        p = os.path.join(runs, "llm_abcd", "preds.jsonl")
        recs = [json.loads(x) for x in open(p)]
        recs[0]["error"] = "HTTPError 500"
        open(p, "w").writelines(json.dumps(r) + "\n" for r in recs)
        rc, g = run()
        check("check_reg: an LLM row with a request error closes that dataset's gate only (exit 1)",
              rc == 1 and not g["abcd"] and g["star"] and g["multiwoz"])
        recs[0]["error"] = None
        open(p, "w").writelines(json.dumps(r) + "\n" for r in recs[1:])
        check("check_reg: a missing LLM row fails", run()[0] == 1)
        open(p, "w").writelines(json.dumps(r) + "\n" for r in recs)
        check("check_reg: restored, every gate open again", run() == (0, {"star": True, "abcd": True, "multiwoz": True}))
        fp = os.path.join(runs, "ft_star", "result.json")
        good_ft = json.load(open(fp))
        json.dump(dict(good_ft, argv=["x.py", "--test-registration", reg, "--dry-run-dev"]), open(fp, "w"))
        check("check_reg: a dry-run FT result fails", run()[0] == 1)
        json.dump(dict(good_ft, lr=3e-05), open(fp, "w"))
        rc, g = run()
        check("check_reg: FT trained with another lr closes STAR's gate", rc == 1 and not g["star"] and g["abcd"])
        json.dump(good_ft, open(fp, "w"))
        tp = os.path.join(runs, "tfidf_multiwoz", "result.json")
        good_tf = json.load(open(tp))
        json.dump(dict(good_tf, C=4.0), open(tp, "w"))
        rc, g = run()
        check("check_reg: TFIDF with another C closes MultiWOZ's gate", rc == 1 and not g["multiwoz"] and g["star"])
        json.dump(good_tf, open(tp, "w"))
        lp = os.path.join(runs, "llm_star", "result.json")
        good_llm = json.load(open(lp))
        bad_argv = [("10" if (j > 0 and good_llm["argv"][j - 1] == "--k") else x) for j, x in enumerate(good_llm["argv"])]
        json.dump(dict(good_llm, argv=bad_argv), open(lp, "w"))
        rc, g = run()
        check("check_reg: an LLM run with --k 10 instead of LLM*'s 50 closes STAR's gate", rc == 1 and not g["star"])
        json.dump(dict(good_llm, argv=good_llm["argv"] + ["--think"]), open(lp, "w"))
        check("check_reg: an LLM run with --think when LLM* has none fails", run()[0] == 1)
        json.dump(good_llm, open(lp, "w"))
        open(qp, "a").write("2026-10-28T01:00:00Z reg_llm_abcd start: cmd:echo llm abcd --k 10 (gate: x)\n")
        rc, g = run()
        check("check_reg: a queue start of reg_llm_abcd with another line closes ABCD's gate", rc == 1 and not g["abcd"])
        open(qp, "w").write("2026-10-28T00:00:00Z queue start: jobs\n" + qlog
                            + "2026-10-28T01:00:00Z reg_llm_abcd start: cmd:echo llm abcd --k 50 (gate: x)\n")
        check("check_reg: a re-queued identical line keeps the gate open", run() == (0, {"star": True, "abcd": True,
                                                                                         "multiwoz": True}))
        recs2 = [dict(r, pred=None) if j == 0 else r for j, r in enumerate(recs)]
        open(p, "w").writelines(json.dumps(r) + "\n" for r in recs2)
        rc, g = run()
        fl = json.load(open(cj))["flags"]
        check("check_reg: 1 invalid answer of 5 raises the 0.5% flag but leaves the gate open",
              rc == 0 and g["abcd"] and fl["abcd: LLM invalid answers at most 0.5% of rows"] is False)


if __name__ == "__main__":
    test_rules()
    test_rows_rule()
    test_draw()
    test_end_to_end()
    test_jobs()
    test_check_reg()
    n_fail = sum(x.startswith("FAIL") for x in OUT)
    print(f"{len(OUT) - n_fail} passed, {n_fail} failed")
    print("PASSED" if n_fail == 0 else "FAILED")
    sys.exit(1 if n_fail else 0)
