"""e81's pick and bar script on made-up rows (plan logs/plan_e81_T_screen.md; added 2026-10-21).

    OMP_NUM_THREADS=1 .venv-dl/bin/python tests/test_e81_pick.py

CPU only; reads no data and writes nothing in the study. Each case gets a temporary folder with a copy of
scripts/e81_pick.py, a frozen record (06:00 UTC), made-up TM rows (results/explore/abcd.jsonl) and LR rows
(results/explore/dryrun/<name>/seed0.json); the copy then runs there as a child process (its ROOT is its own folder's
parent). It asserts:
- the smoke stage: three tiny rows with T 50 and the ABCD rows' dev bits pass; T = clauses (the --T flag lost) or other
  dev bits fail;
- the screen picks: +0.3 or more over T = clauses is taken (also at exactly +0.3), +0.29 is not, an exact tie goes to
  T = clauses, a tie between T 2000 and T 1000 goes to T 2000, and the C4 agent's pick gets no follow-up;
- the follow-up job file: STAR and MultiWOZ at 16,000 clauses with T = r x 16,000 at seeds 1 and 2, ABCD at 4,000 at
  seed 2, bge-large's seed-2 row at T = clauses, and nothing else; a second screen refuses to overwrite it;
- the final stage: B1, B2 (met, not met, and the pick staying at T = clauses) with its last-10 line, the final-epoch and
  last-10 differences and TM minus LR, against values computed here from the made-up rows;
- the refusals: an e81 row written before the freeze, a refined row not on the a3B table, other TM settings, other dev
  bits, two rows with one tag and seed, LR on other bits.
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import e81_pick as P  # noqa: E402

BASE = {"star": 0.65, "abcd": 0.84, "multiwoz": 0.31}
# 4k screen rows, seed 1: accuracy over T = clauses at T 2000 and T 1000
SCREEN = {("a3B", "star"): (0.001, 0.0), ("a3B", "abcd"): (0.0029, 0.003), ("a3B", "multiwoz"): (0.02, 0.0),
          ("bge", "star"): (0.004, 0.004), ("bge", "abcd"): (0.0, 0.0), ("bge", "multiwoz"): (0.001, -0.01),
          ("C4", "star"): (0.0, 0.05), ("C4", "abcd"): (0.0, 0.0), ("C4", "multiwoz"): (0.0, 0.0)}
WANT_PICK = {("a3B", "star"): 4000, ("a3B", "abcd"): 1000, ("a3B", "multiwoz"): 2000, ("bge", "star"): 2000, ("bge", "abcd"): 4000,
             ("bge", "multiwoz"): 4000, ("C4", "star"): 1000, ("C4", "abcd"): 4000, ("C4", "multiwoz"): 4000}
LR_ACC = {"star": 0.66, "abcd": 0.85, "multiwoz": 0.36}


def hist(acc: float, k: int) -> list:
    """20 evaluations whose best is acc (the row's best-dev snapshot), with a final value and last-10 mean that vary by k."""
    v = [acc - 0.001 * ((3 * i + k) % 7) for i in range(20)]
    v[k % 20] = acc
    return [{"epoch": 2 * (i + 1), "val_acc": x} for i, x in enumerate(v)]


def row(tag, seed, agent, ds, c, T, acc, k=0, time="2026-10-21T07:00:00Z", **bad):
    a = P.AGENTS[agent]
    cfg = dict(P.BASE_CFG, n_clauses=c, T=T, s=float(a["s"][ds]), seed=seed)
    cfg.update(bad.get("cfg", {}))
    table = bad.get("table", P.A3B_SHA if agent == "a3B" else "c4-or-none")
    return {"tag": tag, "seed": seed, "arm": "tm", "dataset": ds, "encoder": a["enc"], "parts": a["parts"], "lex_k": 1000, "init": "half",
            "cfg": cfg, "acc": acc, "time": time, "static_info": {p: {"table_file_sha256": table} for p in ("xcust", "xrecent")},
            "bits_sha256": {"dev": bad.get("bits", f"bits-{agent}-{ds}")}, "history": hist(acc, k)}


def stage_a_rows(screen: dict = SCREEN) -> dict:
    """(tag, seed) -> row: the T = clauses references (written before the freeze) and the 4k screens (after it)."""
    R = {}
    k = 0
    for agent, a in P.AGENTS.items():
        for ds in ("star", "abcd", "multiwoz"):
            k += 1
            R[(a["ref4k"][ds], 1)] = row(a["ref4k"][ds], 1, agent, ds, 4000, 4000, BASE[ds], k, time="2026-10-20T00:00:00Z")
            for T, bump in zip((2000, 1000), screen[(agent, ds)]):
                t = P.tag(ds, agent, 4000, a["s"][ds], T)
                R[(t, 1)] = row(t, 1, agent, ds, 4000, T, BASE[ds] + bump, k + T // 1000)
            for i, (t, sd) in enumerate(a["reffull"][ds]):
                if t.startswith("e81_") or (t, sd) in R:
                    continue
                c = P.FULL[ds]
                acc = BASE[ds] + 0.01 + 0.002 * i
                if (agent, ds) == ("a3B", "multiwoz"):
                    acc = (0.3122706889523033, 0.3092811523304797)[sd - 1]  # e82's Stage A values (B2's constants)
                R[(t, sd)] = row(t, sd, agent, ds, c, c, acc, k + 5 + i, time="2026-10-20T00:00:00Z")
    return R


def stage_b_rows(b2_met: bool) -> dict:
    m = (0.340, 0.343) if b2_met else (0.330, 0.333)
    R = {}
    for t, sd, agent, ds, c, T, acc, k in [
            (P.tag("abcd", "a3B", 4000, 16, 1000), 2, "a3B", "abcd", 4000, 1000, 0.851, 31),
            (P.tag("multiwoz", "a3B", 16000, 8, 8000), 1, "a3B", "multiwoz", 16000, 8000, m[0], 32),
            (P.tag("multiwoz", "a3B", 16000, 8, 8000), 2, "a3B", "multiwoz", 16000, 8000, m[1], 33),
            (P.tag("star", "bge", 16000, 16, 8000), 1, "bge", "star", 16000, 8000, 0.664, 34),
            (P.tag("star", "bge", 16000, 16, 8000), 2, "bge", "star", 16000, 8000, 0.662, 35),
            (P.tag("star", "bge", 16000, 16, 16000), 2, "bge", "star", 16000, 16000, 0.659, 36)]:
        R[(t, sd)] = row(t, sd, agent, ds, c, T, acc, k)
    return R


def write(tmp: Path, rows: dict, lr_bits_bad: bool = False) -> None:
    (tmp / "scripts").mkdir(parents=True, exist_ok=True)
    (tmp / "logs").mkdir(exist_ok=True)
    (tmp / "results" / "explore").mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "scripts" / "e81_pick.py", tmp / "scripts" / "e81_pick.py")
    (tmp / "logs" / "plan_e81_T_screen.frozen.json").write_text(json.dumps({"frozen_utc": "2026-10-21T06:00:00Z"}))
    lines = []
    for v in rows.values():
        lines += v if isinstance(v, list) else [v]
    (tmp / "results" / "explore" / "abcd.jsonl").write_text("".join(json.dumps(d) + "\n" for d in lines))
    for (agent, ds), name in P.LR.items():
        p = tmp / "results" / "explore" / "dryrun" / name
        p.mkdir(parents=True, exist_ok=True)
        acc = P.B2_LR if (agent, ds) == ("a3B", "multiwoz") else LR_ACC[ds]
        bits = "other-bits" if lr_bits_bad and (agent, ds) == ("bge", "star") else f"bits-{agent}-{ds}"
        (p / "seed0.json").write_text(json.dumps({"arm": "lr", "dev_acc": acc, "bits_sha256": {"dev": bits}, "grid_c": P.GRID_C}))


def run(tmp: Path, stage: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-I", str(tmp / "scripts" / "e81_pick.py"), "--stage", stage], cwd=tmp, capture_output=True, text=True)


def three(d: dict) -> tuple:
    v = [x["val_acc"] for x in d["history"]]
    return d["acc"], v[-1], sum(v[-10:]) / 10


def b1_line(R: dict, agent: str, ds: str, T: int) -> str:
    """The final stage's line for a follow-up pick, computed here from the made-up rows."""
    a, c = P.AGENTS[agent], P.FULL[ds]
    s = a["s"][ds]
    if c == 16000:
        new = [R[(P.tag(ds, agent, c, s, T), sd)] for sd in (1, 2)]
    else:
        new = [R[(P.tag(ds, agent, c, s, T), 1)], R[(P.tag(ds, agent, c, s, T), 2)]]
    ref = [R[(t, sd)] for t, sd in a["reffull"][ds]]
    en, er = [three(d) for d in new], [three(d) for d in ref]
    diff = [100 * ((en[0][i] + en[1][i]) / 2 - (er[0][i] + er[1][i]) / 2) for i in range(3)]
    lr = P.B2_LR if (agent, ds) == ("a3B", "multiwoz") else LR_ACC[ds]
    tm = (new[0]["acc"] + new[1]["acc"]) / 2
    return (f"{agent:4} {ds:8}: T {T} against T {c} at {c} clauses, mean of seeds 1-2: {diff[0]:+.2f} "
            f"(B1 needs >= +0.5: {'MET' if diff[0] >= 0.5 - 1e-9 else 'not met'}); beside it: final epoch {diff[1]:+.2f}, "
            f"last-10 mean {diff[2]:+.2f}; TM minus LR {100 * (tm - lr):+.2f}")


def expect_fail(name: str, rows: dict, stage: str, msg: str, lr_bits_bad: bool = False, extra=None) -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        write(tmp, rows, lr_bits_bad)
        if extra:
            extra(tmp)
        p = run(tmp, stage)
        assert p.returncode != 0 and msg in p.stderr, (name, p.returncode, p.stdout[-500:], p.stderr[-500:])
        print(f"ok refusal: {name} ({p.stderr.strip().splitlines()[-1]})")


def smoke_rows(T: int = 50, bits_bad: str = "") -> dict:
    out = {}
    for agent in P.AGENTS:
        kw = {"bits": "bits-other"} if agent == bits_bad else {}
        out[(f"e81smoke_{agent}", 1)] = row(f"e81smoke_{agent}", 1, agent, "abcd", 200, T, 0.5, 1, cfg={"epochs": 2}, **kw)
    return out


def main() -> None:
    A = stage_a_rows()
    for name, rows, code, want in (("three tiny rows", smoke_rows(), 0, ""), ("T = clauses (the --T flag lost)", smoke_rows(T=200), 1, "smoke: 3 of 3 failed"),
                                   ("other dev bits for bge-large", smoke_rows(bits_bad="bge"), 1, "smoke: 1 of 3 failed")):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            write(tmp, {**A, **rows})
            p = run(tmp, "smoke")
            assert (p.returncode == 0) == (code == 0) and want in p.stderr, (name, p.stdout, p.stderr)
            assert p.stdout.count(": ok (") == (3 if code == 0 else 3 - int(want.split()[1])), (name, p.stdout)
            print(f"ok smoke stage: {name} -> {'passes' if code == 0 else 'fails'}")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        write(tmp, A)
        p = run(tmp, "screen")
        assert p.returncode == 0, p.stderr
        for (agent, ds), T in WANT_PICK.items():
            assert any(ln.startswith(f"{agent:4} {ds:8} 4k") and ln.endswith(f"; pick T {T}") for ln in p.stdout.splitlines()), (agent, ds, p.stdout)
        print("ok screen picks: +0.3 taken (also exactly +0.3), +0.29 not, exact tie to T = clauses, T 2000 / T 1000 tie to T 2000")
        jobs = (tmp / "logs" / "jobs_e81b.txt").read_text().splitlines()
        want = [P.job("abcd", "a3B", 4000, 16, 1000, 2), P.job("multiwoz", "a3B", 16000, 8, 8000, 1), P.job("multiwoz", "a3B", 16000, 8, 8000, 2),
                P.job("star", "bge", 16000, 16, 8000, 1), P.job("star", "bge", 16000, 16, 8000, 2), P.job("star", "bge", 16000, 16, 16000, 2)]
        assert jobs[0].startswith("# e81b") and sorted(jobs[1:]) == sorted(want), jobs
        print(f"ok follow-up jobs: exactly the {len(want)} expected lines (no C4 line although C4 STAR picked T 1000)")
        p2 = run(tmp, "screen")
        assert p2.returncode != 0 and "exists; not overwriting" in p2.stderr, p2.stderr
        print("ok a second screen refuses to overwrite jobs_e81b.txt")
        for b2_met in (True, False):
            R = {**A, **stage_b_rows(b2_met)}
            write(tmp, R)
            p = run(tmp, "final")
            assert p.returncode == 0, p.stderr
            out = p.stdout.splitlines()
            for agent, ds, T in (("a3B", "abcd", 1000), ("a3B", "multiwoz", 8000), ("bge", "star", 8000)):
                ln = b1_line(R, agent, ds, T)
                assert ln in out, (ln, p.stdout)
            tm = (R[(P.tag("multiwoz", "a3B", 16000, 8, 8000), 1)]["acc"] + R[(P.tag("multiwoz", "a3B", 16000, 8, 8000), 2)]["acc"]) / 2
            today = 100 * (P.B2_TODAY - P.B2_LR)
            b2 = (f"B2 (refined agent, MultiWOZ): TM minus LR {100 * (tm - P.B2_LR):+.2f} (today {today:+.2f}; needs >= {today / 2:+.2f}): "
                  f"{'MET' if b2_met else 'not met'}")
            assert out[-2] == b2, (out[-2], b2)
            old_rows = [R[(t, sd)] for t, sd in P.AGENTS["a3B"]["reffull"]["multiwoz"]]
            new_rows = [R[(P.tag("multiwoz", "a3B", 16000, 8, 8000), sd)] for sd in (1, 2)]
            l10 = [100 * ((three(a)[2] + three(b)[2]) / 2 - P.B2_LR) for a, b in (old_rows, new_rows)]
            beside = (f"B2 beside it (no rule): last-10 mean TM minus LR {l10[1]:+.2f} (today {l10[0]:+.2f}; half of it {l10[0] / 2:+.2f}): "
                      f"{'steady' if l10[1] >= l10[0] / 2 - 1e-9 else 'not steady'}")
            assert out[-1] == beside, (out[-1], beside)
            assert abs(today - (-5.354)) < 0.001, today
            c4 = [d["acc"] for t, sd in P.AGENTS["C4"]["reffull"]["star"] for d in [R[(t, sd)]]]
            ln = (f"C4   star    : pick T 1000 (no follow-up); TM at T = clauses, 16000 clauses, mean of 2 seed(s) {sum(c4) / 2:.4f}; "
                  f"TM minus LR {100 * (sum(c4) / 2 - LR_ACC['star']):+.2f}")
            assert ln in out, (ln, p.stdout)
            ln = f"bge  abcd    : pick T 4000 (T = clauses); TM at T = clauses, 4000 clauses, mean of 1 seed(s) {BASE['abcd']:.4f}; TM minus LR {100 * (BASE['abcd'] - LR_ACC['abcd']):+.2f}"
            assert ln in out, (ln, p.stdout)
            print(f"ok final stage (B2 {'met' if b2_met else 'not met'}): B1 lines with the epoch columns, TM minus LR, the r = 1 lines and B2 equal values computed here")
    # B2 when the refined agent's MultiWOZ pick stays at T = clauses: today's TM stands, not met, not steady
    A1 = stage_a_rows({**SCREEN, ("a3B", "multiwoz"): (0.0, 0.0)})
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        write(tmp, {**A1, **stage_b_rows(True)})
        p = run(tmp, "final")
        assert p.returncode == 0, p.stderr
        out = p.stdout.splitlines()
        old_rows = [A1[(t, sd)] for t, sd in P.AGENTS["a3B"]["reffull"]["multiwoz"]]
        today = 100 * (P.B2_TODAY - P.B2_LR)
        l10 = 100 * ((three(old_rows[0])[2] + three(old_rows[1])[2]) / 2 - P.B2_LR)
        want = [f"a3B  multiwoz: pick T 4000 (T = clauses); TM at T = clauses, 16000 clauses, mean of 2 seed(s) {P.B2_TODAY:.4f}; TM minus LR {today:+.2f}",
                f"B2 (refined agent, MultiWOZ): TM minus LR {today:+.2f} (today {today:+.2f}; needs >= {today / 2:+.2f}): not met",
                f"B2 beside it (no rule): last-10 mean TM minus LR {l10:+.2f} (today {l10:+.2f}; half of it {l10 / 2:+.2f}): not steady"]
        assert want[0] in out and out[-2:] == want[1:], (want, p.stdout)
        print("ok final stage (refined MultiWOZ pick T = clauses): today's TM stands, B2 not met, not steady")
    # refusals
    t_screen = P.tag("star", "a3B", 4000, 16, 2000)
    B = dict(A)
    B[(t_screen, 1)] = dict(A[(t_screen, 1)], time="2026-10-21T05:59:59Z")
    expect_fail("e81 row written before the freeze", B, "screen", "written before the freeze")
    B = dict(A)
    B[(t_screen, 1)] = row(t_screen, 1, "a3B", "star", 4000, 2000, 0.651, table="not-a3B")
    expect_fail("refined row not on the a3B table", B, "screen", "not on the a3B table")
    B = dict(A)
    B[(t_screen, 1)] = row(t_screen, 1, "a3B", "star", 4000, 2000, 0.651, cfg={"epochs": 30})
    expect_fail("other TM settings (30 epochs)", B, "screen", "other TM settings differ")
    B = dict(A)
    B[(t_screen, 1)] = row(t_screen, 1, "a3B", "star", 4000, 2000, 0.651, bits="bits-other")
    expect_fail("other dev bits", B, "screen", "dev bits")
    B = dict(A)
    B[(t_screen, 1)] = [A[(t_screen, 1)], A[(t_screen, 1)]]
    expect_fail("two rows with one tag and seed", B, "screen", "2 rows (need exactly 1)")
    expect_fail("LR on other bits", {**A, **stage_b_rows(True)}, "final", "not LR on the same dev bits", lr_bits_bad=True)
    print("PASSED")


if __name__ == "__main__":
    main()
