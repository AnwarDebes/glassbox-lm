#!/usr/bin/env python
"""R25 gate: check the finished test runs against the freeze, before the report is read.

    .venv-dl/bin/python scripts/r25_check.py | tee logs/r25_check.out

Read only, except one temporary file per report rerun (outside the study). One line per check; stops at the first
failure. Written before any R25 test number (adapted from scripts/r23_check.py; the checks listed by the review of
2026-10-21, project notes, not published). S_STAR, S_ABCD and DATASETS are set at the freeze from
scripts/e82_pick.py --stage final and the go-bars; the script refuses to run with them unset. Checks:
- the registration text is unchanged since the freeze; every source on R25's path is unchanged since the freeze, and
  other changes are only this checker;
- since R23's freeze, abcd_confirm.py differs by exactly the declared line (the "multiwoz" --dataset choice);
- the a3B table's six files have their registered hashes;
- the seed files are exactly the jobs of logs/jobs_R25.txt, written after the freeze, and the new arms' xcust and
  xrecent were built on the a3B table file (the literal hash);
- cfg: the new arm equals the C4 reference's but for s, seed and the table; mem and words bit names are identical;
- bits: train and dev equal the dry runs' and e82-R3's row at s*; one test hash per arm;
- seed 1's dev accuracy equals e82-R3's row at s* exactly;
- each npz gives its json's test accuracy; y, convo_id, k, w_bert and w_roberta are equal in every arm and reference;
  the test example files have their registered hashes;
- the copied references: each file equals the list and its source; the folders hold exactly the listed files; their
  test-bit hashes are the registered ones; report_abcd.py rerun on R22star and R23abcd gives their reports byte for
  byte.
"""

from __future__ import annotations

import calendar
import difflib
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
CONF = ROOT / "results" / "confirm"
FROZEN = ROOT / "registrations" / "R25_refined_table_agent.frozen.json"
R23_FROZEN = ROOT / "registrations" / "R23_tuned_s_abcd.frozen.json"
S_STAR = None  # set at the freeze (e82_pick.py --stage final)
S_ABCD = None
DATASETS = None  # set at the freeze: the datasets whose go-bars passed, e.g. ("star", "abcd")
TABLE = {"table.npz": "4a1ed2aed7aa4ce726892446a2cdf34b23a6e43d3c2e943f0fb46c5210d7d855",
         "words.txt": "49dae3df4202d9769e377e8ecfcd4adeffc799907745cfbcbdca83cb7ccd0a29",
         "tokenize_ref.py": "2c3c1aff44ed34e017276221dff31ea29e7f9ab1f8f3a77cd0f193ee36e91803",
         "vectors.npy": "21d0c4fb148498531a6523f9635401a6215ef4dd5c8cd4b7ca8ce6e7b80f8e14",
         "README.md": "35ae923e408084ff8d9512a06eea5f481aff0037da0e23d06b111286e6dbd85f",
         "MANIFEST.sha256": "1d1a00ad74634f77d38894e4a476c1717494c94a76450210518ec9bc8ed267f3"}
EXAMPLES_TEST = {"star": "f9cc8f150b25d94da38930e55e4316f68cac3420da58d07fc67f9990853ed556",
                 "abcd": "190312c67b2258b4353b4f7a95bcd8edff6ddc1798e81ece9a3fa0d13d930c4d"}
ON_PATH = ["src/dl/count_text.py", "scripts/table_run.py", "scripts/newtable_run.py", "scripts/seq_queue.sh", "scripts/abcd_confirm.py",
           "scripts/report_abcd.py", "scripts/report_twolevel.py", "src/dl/static_text.py", "src/dl/abcd.py", "src/dl/tm.py",
           "src/dl/tm_init.py", "src/dl/boolean.py", "src/dl/linear.py"]
ALLOWED = {"scripts/r25_check.py"}
CONFIRM_LINE = ('    ap.add_argument("--dataset", default="abcd", choices=["abcd", "star", "multiwoz"], help="star: scripts/star_prepare.py '
                'examples (added 2026-10-17 14:18); multiwoz: scripts/multiwoz_prepare.py (added 2026-10-19 for e77)")')
REFS = {"star": {"reg": "R25star", "c4": ("tm_mem_c4_s16_R22", "R22star", "tm_mem_c4_s16", "d3f682f2458723e1ca2037b6f4e0722fe58e93d58adcc7a78f1f47934755ab80"),
                 "bge": ("tm_mem_bge_s16_R22", "R22star", "tm_mem_bge_s16", "05f28f952a0414ce4b8c8ec5013718da5c106cc81694debbfdddad6024a2c03e"),
                 "clauses": 16000, "n_conv": 611},
        "abcd": {"reg": "R25abcd", "c4": ("tm_mem_c4_s16_R23", "R23abcd", "tm_mem_c4_s16", "fb13341b00177aab9462c43a0c72cea3b5cf11618ce4591fd87715cced36d416"),
                 "bge": ("tm_mem_bge_s24_R23", "R23abcd", "tm_mem_bge_s24", "0fc386cd4bee842663b21c050c90970e8e63f703845beadb05fdefb3050fda74"),
                 "clauses": 4000, "n_conv": 1004}}
REPORTS = {"R22star": ("ca2f182805401a9db27f5aa3ae47d46936f342764b1871555e314d6f52b5e7a6",
                       ["--arms", "tm_mem_c4_s16", "tm_mem_bge_s16", "tm_mem_c4_s8_R21", "lr_mem_c4_R21", "tm_mem_bge_s8_R18", "--pairs",
                        "tm_mem_c4_s16:tm_mem_c4_s8_R21", "tm_mem_c4_s16:lr_mem_c4_R21", "tm_mem_c4_s16:tm_mem_bge_s16",
                        "tm_mem_bge_s16:tm_mem_bge_s8_R18", "tm_mem_c4_s16:tm_mem_bge_s8_R18"], "report_R22star.json"),
           "R23abcd": ("3c6c0c1d1015bc7eb778ed5f0ff74e8f6ba4fa57c3a8bd390acb7345ddd56921",
                       ["--arms", "tm_mem_c4_s16", "tm_mem_bge_s24", "tm_mem_c4_s8_R21", "tm_mem_bge_s8_R9b", "--pairs",
                        "tm_mem_c4_s16:tm_mem_bge_s24", "tm_mem_c4_s16:tm_mem_c4_s8_R21", "tm_mem_bge_s24:tm_mem_bge_s8_R9b"], "report_R23abcd.json")}
META = ("y", "convo_id", "k", "w_bert", "w_roberta")
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731


def need(cond: bool, msg: str) -> None:
    print(("ok   " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        sys.exit(1)


def main() -> None:
    if S_STAR is None or S_ABCD is None or DATASETS is None:
        raise SystemExit("set S_STAR, S_ABCD and DATASETS at the freeze")
    import e82_pick

    s_of = {"star": S_STAR, "abcd": S_ABCD}
    rec = json.loads(FROZEN.read_text())
    t_freeze = calendar.timegm(time.strptime(rec["frozen_utc"], "%Y-%m-%dT%H:%M:%SZ"))
    need(sha(ROOT / rec["registration"]) == rec["sha256"], f"registration text unchanged ({rec['sha256'][:16]})")
    now = {str(p.relative_to(ROOT)): sha(p) for d in ("src", "scripts") for p in sorted((ROOT / d).rglob("*.py")) + sorted((ROOT / d).rglob("*.sh"))}
    for f in ON_PATH:
        need(now.get(f) == rec["sources"].get(f) is not None, f"on R25's path, unchanged since the freeze: {f} ({now[f][:8]})")
    changed = sorted(f for f in set(now) | set(rec["sources"]) if rec["sources"].get(f) != now.get(f))
    need(set(changed) <= ALLOWED, f"other changes since the freeze: {changed}")
    r23 = json.loads(R23_FROZEN.read_text())["sources"]["scripts/abcd_confirm.py"]
    pristine = ROOT / "logs" / "pre_e77" / "abcd_confirm.py"
    need(sha(pristine) == r23, f"logs/pre_e77/abcd_confirm.py is R23's frozen abcd_confirm.py ({r23[:8]})")
    diff = [l for l in difflib.unified_diff(pristine.read_text().splitlines(), (ROOT / "scripts" / "abcd_confirm.py").read_text().splitlines(), lineterm="", n=0)
            if l[:1] in "+-" and not l.startswith(("+++", "---"))]
    need(len(diff) == 2 and diff[0].startswith("-") and diff[1] == "+" + CONFIRM_LINE, "since R23's freeze, abcd_confirm.py differs by exactly the declared --dataset line")
    tdir = ROOT / "data" / "tables" / "c4words_a3B_v1"
    need(all(sha(tdir / f) == h for f, h in TABLE.items()), "the a3B table's six files have their registered hashes")
    for ds in ("star", "abcd"):
        need(sha(ROOT / "data" / ds / "examples_test.json.gz") == EXAMPLES_TEST[ds], f"{ds} test examples {EXAMPLES_TEST[ds][:8]}")

    jobs = {}
    for line in (ROOT / "logs" / "jobs_R25.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        arg = lambda k: (re.search(rf"--{k} (\S+)", line) or [None, None])[1]  # noqa: E731
        jobs[(arg("registration"), arg("name"), int(arg("seeds")))] = {"s": float(arg("s")), "clauses": int(arg("clauses")), "dataset": arg("dataset")}
    found = {(p.parent.parent.name, p.parent.name, int(p.stem[4:])) for ds in DATASETS
             for p in (CONF / REFS[ds]["reg"] / f"tm_mem_a3B_s{s_of[ds]}").glob("seed*.json")}
    need(len(jobs) == 5 * len(DATASETS) and found == set(jobs), f"seed files are exactly the {len(jobs)} jobs of logs/jobs_R25.txt")

    rows = {}
    for line in (ROOT / "results" / "explore" / "abcd.jsonl").read_text().splitlines():
        d = json.loads(line)
        if d.get("arm") == "tm" and "tag" in d:
            rows[(d["tag"], d["seed"])] = d
    for ds in DATASETS:
        R = REFS[ds]
        name = f"tm_mem_a3B_s{s_of[ds]}"
        folder = CONF / R["reg"] / name
        c4ref = json.loads((CONF / R["c4"][1] / R["c4"][2] / "seed1.json").read_text())
        e82row = rows[(e82_pick.tag(ds, R["clauses"], s_of[ds]), 1)]
        dry = json.loads((ROOT / "results" / "explore" / "dryrun" / f"dry_{R['reg']}_{name}" / "seed1.json").read_text())
        names_new = json.loads((folder / "bit_names.json").read_text())
        names_c4 = json.loads((CONF / R["c4"][1] / R["c4"][2] / "bit_names.json").read_text())
        k_mem_words = sum(1 for n in names_c4 if not n.startswith("some word of"))
        need(names_new[:k_mem_words] == names_c4[:k_mem_words], f"{ds}: mem and words bit names identical to the C4 reference's ({k_mem_words})")
        test_bits, meta0 = set(), None
        for seed in range(1, 6):
            j = json.loads((folder / f"seed{seed}.json").read_text())
            tag = f"{R['reg']}/{name}/seed{seed}"
            need(j["registration"] == R["reg"] and j["name"] == name and j["seed"] == seed and j["arm"] == "tm" and j["encoder"] == "none"
                 and j["parts"] == ["mem", "words", "xcust", "xrecent"], f"{tag}: settings as the job line")
            need(all(v["table_file_sha256"] == TABLE["table.npz"] for k, v in j["static_info"].items() if k in ("xcust", "xrecent")),
                 f"{tag}: xcust and xrecent on the a3B table file ({TABLE['table.npz'][:16]})")
            t_json = calendar.timegm(time.strptime(j["time"], "%Y-%m-%dT%H:%M:%SZ"))
            need(t_json > t_freeze and min(p.stat().st_mtime for p in folder.iterdir()) > t_freeze, f"{tag}: written {j['time']}, after the freeze")
            need({**j["cfg"], "seed": None, "s": None} == {**c4ref["cfg"], "seed": None, "s": None} and j["cfg"]["s"] == float(s_of[ds])
                 and j["cfg"]["seed"] == seed and j["cfg"]["n_clauses"] == R["clauses"], f"{tag}: cfg equals the C4 reference's but s {s_of[ds]} and the seed")
            need(j["bits_sha256"]["train"] == dry["bits_sha256"]["train"] == e82row["bits_sha256"]["train"]
                 and j["bits_sha256"]["dev"] == dry["bits_sha256"]["dev"] == e82row["bits_sha256"]["dev"], f"{tag}: train/dev bits equal the dry run's and e82-R3's row")
            if seed == 1:
                need(j["dev_acc"] == e82row["acc"], f"{tag}: dev accuracy {j['dev_acc']:.6f} equals e82-R3's row exactly")
            z = np.load(folder / f"seed{seed}.npz")
            need(abs(float((z["pred"] == z["y"]).mean()) - j["test_acc"]) < 1e-12, f"{tag}: npz gives the json's test accuracy")
            if meta0 is None:
                meta0 = {k: z[k] for k in META}
            need(all(np.array_equal(z[k], meta0[k]) for k in META), f"{tag}: the five meta arrays equal the other arms'")
            test_bits.add(j["bits_sha256"]["test"])
        need(len(test_bits) == 1, f"{ds}: one test-bit hash over the new arm's seeds")
        for key in ("c4", "bge"):
            dst, src_reg, src, test_hash = R[key]
            for p in sorted((CONF / R["reg"] / dst).glob("seed*.npz")):
                z = np.load(p)
                need(all(np.array_equal(z[k], meta0[k]) for k in META), f"reference {dst}/{p.name}: the five meta arrays equal the new arm's")
            j1 = json.loads((CONF / R["reg"] / dst / "seed1.json").read_text())
            need(j1["bits_sha256"]["test"] == test_hash, f"reference {dst}: test bits {test_hash[:8]}")
        need(len(np.unique(meta0["convo_id"])) == R["n_conv"], f"{ds}: {len(meta0['y'])} test turns in {R['n_conv']} conversations")
    listed = {}
    for line in (ROOT / "logs" / "r25_refs.out").read_text().splitlines():
        path, h, src = line.split(" ", 2)
        listed[path] = (h, src.strip("()").removeprefix("from "))
    for path, (h, src) in listed.items():
        dst = CONF / path
        need(sha(dst) == h and sha(CONF / src / dst.name) == h, f"copy {path} equals the list and its source {src}/{dst.name}")
    present = {str(p.relative_to(CONF)) for ds in DATASETS for p in (CONF / REFS[ds]["reg"]).glob("*_R2[23]/*")}
    need(present == set(listed), f"the reference folders hold exactly the {len(listed)} listed files")
    scratch = Path("/tmp") / f"r25_check_reports_{int(time.time())}"
    scratch.mkdir()
    for reg, (h, argv, name) in REPORTS.items():
        out = scratch / name
        subprocess.run([sys.executable, str(ROOT / "scripts" / "report_abcd.py"), "--registration", reg, *argv, "--out", str(out)],
                       check=True, capture_output=True)
        need(sha(out) == h == sha(CONF / reg / name), f"report_abcd.py rerun on {reg} gives {name} byte for byte ({h[:8]})")
    print("all checks passed", flush=True)


if __name__ == "__main__":
    main()
