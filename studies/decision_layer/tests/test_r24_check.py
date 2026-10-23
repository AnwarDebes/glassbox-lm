"""scripts/r24_check.py, full mode and --gate, on made-up files (registrations/R24_multiwoz_random_negatives.md; added
2026-10-22, before the freeze and before any MultiWOZ test number).

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 .venv-dl/bin/python tests/test_r24_check.py

CPU only; nothing written in the study. Each case builds a temporary study folder: a copy of scripts/r24_check.py, a
stub for every source on R24's path, a registration and its freeze record (an hour ago), e85's s* line, the 16k seed-1
dev rows, the dry runs, the 18 seed files of logs/jobs_R24.txt (scripts/e85_pick.py's r24_jobs; copies of e85_pick.py
and e84_check.py), the test and train examples, and the bge-large file with its dev stage (different provenance entries;
the copied checker's DEVSTAGE hash is set to the made-up dev-stage file's). The refined agent is at s* 16, so its dev
row is e85b's. It asserts:
- the full mode passes on consistent files;
- one refusal per check (each case changes one thing and must fail on that check's line), including a test turn of
  examples_test.json.gz missing from the arms; the dialogue count comes from the examples, not a constant;
- --gate: one agent missing the gate keeps exit 0 and says so; both missing gives exit 3; exactly -1.0 holds.
"""

import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import e85_pick as E  # noqa: E402
import r24_check as R  # noqa: E402

SSTAR = {"refined": 16, "bge-large": 8, "C4": 8}
ACC = {"ref": 0.36, "c4": 0.35, "bge": 0.345}
LR = {"ref": (0.364316, 0.001), "c4": (0.358609, 0.001), "bge": (0.365946, 0.01)}
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
stamp = lambda t: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))  # noqa: E731


def h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


class Case:
    def __init__(self, acc: dict = ACC, dialogues: int = 1000) -> None:
        t = self.t = Path(tempfile.mkdtemp())
        self.freeze = time.time() - 3600
        (t / "data/emb").mkdir(parents=True)
        arr = {"train__cust": np.ones((3, 2)), "dev__cust": np.zeros((2, 2))}
        np.savez(t / "data/emb/multiwoz__bge-large.devstage.npz", provenance=np.array("dev stage"), **arr)
        np.savez(t / "data/emb/multiwoz__bge-large.npz", provenance=np.array("with test"), **arr, test__cust=np.ones((4, 2)))
        (t / "scripts").mkdir()
        for f in ("e85_pick.py", "e84_check.py"):
            shutil.copy(ROOT / "scripts" / f, t / "scripts" / f)
        src = (ROOT / "scripts" / "r24_check.py").read_text()
        assert src.count(R.DEVSTAGE) == 1
        (t / "scripts" / "r24_check.py").write_text(src.replace(R.DEVSTAGE, sha(t / "data/emb/multiwoz__bge-large.devstage.npz")))
        for f in R.ON_PATH:
            (t / f).parent.mkdir(parents=True, exist_ok=True)
            (t / f).write_text(f"# stub {f}\n")
        reg = t / "registrations" / "R24_multiwoz_random_negatives.md"
        reg.parent.mkdir()
        reg.write_text("R24 stub\n")
        src = {str(p.relative_to(t)): sha(p) for d in ("src", "scripts") for p in sorted((t / d).rglob("*.py")) + sorted((t / d).rglob("*.sh"))}
        (reg.parent / "R24_multiwoz_random_negatives.frozen.json").write_text(json.dumps(
            {"registration": "registrations/R24_multiwoz_random_negatives.md", "sha256": sha(reg), "frozen_utc": stamp(self.freeze), "sources": src}))
        (t / "logs").mkdir()
        (t / "logs" / "e85_pick_final.out").write_text("refined  : ...\ns*: " + ", ".join(f"{a} {s}" for a, s in SSTAR.items()) + "\n")
        (t / "logs" / "jobs_R24.txt").write_text("# jobs_R24\n" + "\n".join(E.r24_jobs(SSTAR)) + "\n")
        rng = np.random.default_rng(0)
        n = 2 * dialogues
        y, cid = rng.integers(0, 5, n), np.arange(n) // 2
        dev_rows, now = [], stamp(time.time())
        for k, a in R.ARMS.items():
            s = SSTAR[a["e85"]]
            tag = a["dev_row"] if s == 8 else f"e85_multiwoz_{a['short']}_c16000_s{s}_nrand"
            cfg = {"n_clauses": 16000, "T": 16000, "s": float(s), "seed": 1, "epochs": 40, "neg_sampling": "random", "lr_rate": 1}
            bits = {"train": h(k + "train"), "dev": h(k + "dev")}
            info = {"static_info": {x: {"table_file_sha256": a["table"]} for x in ("xcust", "xrecent")}} if a["table"] else {}
            dev_rows.append({"tag": tag, "arm": "tm", "seed": 1, "init": "half", "acc": acc[k], "cfg": cfg, "bits_sha256": bits, "time": "2026-10-22T12:00:00Z"})
            dry = t / "results/explore/dryrun" / f"dry_R24mwoz_{a['tm']}_s8"
            dry.mkdir(parents=True)
            (dry / "seed1.json").write_text(json.dumps({"bits_sha256": {**bits, "test": bits["dev"]}}))
            (t / "results/explore/dryrun" / a["lr_dev"]).mkdir(parents=True)
            (t / "results/explore/dryrun" / a["lr_dev"] / "seed0.json").write_text(json.dumps({"dev_acc": LR[k][0], "C": LR[k][1]}))
            for name, seeds in ((a["tm"], range(1, 6)), (a["lr"], [0])):
                folder = t / "results/confirm/R24mwoz" / name
                folder.mkdir(parents=True)
                for sd in seeds:
                    pred = np.where(rng.random(n) < 0.35, y, (y + 1) % 5)
                    np.savez(folder / f"seed{sd}.npz", pred=pred, y=y, convo_id=cid, k=cid % 7, w_bert=cid % 3, w_roberta=cid % 5, S=np.zeros((n, 2)))
                    j = {"registration": "R24mwoz", "name": name, "seed": sd, "time": now, "test_acc": float((pred == y).mean()),
                         "bits_sha256": {**bits, "test": h(k + "test")}}
                    if sd:
                        j.update(arm="tm", init="half", parts=a["parts"], encoder=a["encoder"], cfg={**cfg, "seed": sd}, dev_acc=acc[k] if sd == 1 else 0.3, **info)
                    else:
                        j.update(arm="lr", dev_acc=LR[k][0], C=LR[k][1])
                    (folder / f"seed{sd}.json").write_text(json.dumps(j))
        (t / "results/explore/abcd.jsonl").write_text("".join(json.dumps(d) + "\n" for d in dev_rows))
        (t / "data/multiwoz").mkdir(parents=True)
        for split in ("train", "test"):
            with gzip.open(t / "data/multiwoz" / f"examples_{split}.json.gz", "wt") as f:
                json.dump({"rows": [{"convo_id": int(c)} for c in cid] if split == "test" else [], "buttons": ["a", "b"]}, f)

    def js(self, name: str, sd: int) -> Path:
        return self.t / "results/confirm/R24mwoz" / name / f"seed{sd}.json"

    def edit(self, name: str, sd: int, fn) -> None:
        p = self.js(name, sd)
        j = json.loads(p.read_text())
        fn(j)
        p.write_text(json.dumps(j))

    def run(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, str(self.t / "scripts" / "r24_check.py"), *args], capture_output=True, text=True, cwd=self.t,
                              env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})


def npz_set(c: Case, name: str, sd: int, key: str) -> None:
    p = c.t / "results/confirm/R24mwoz" / name / f"seed{sd}.npz"
    z = dict(np.load(p))
    z[key] = (z[key] + 1) % 5
    np.savez(p, **z)
    c.edit(name, sd, lambda j: j.update(test_acc=float((z["pred"] == z["y"]).mean())))


def job_s12(c: Case) -> None:
    p = c.t / "logs/jobs_R24.txt"
    p.write_text(p.read_text().replace("--s 16 --name tm_mem_ref_nrand --clauses 16000 --seeds 3", "--s 12 --name tm_mem_ref_nrand --clauses 16000 --seeds 3"))


def old(p: Path, c: Case) -> None:
    os.utime(p, (c.freeze - 60, c.freeze - 60))


REFUSALS = [
    ("registration text unchanged", lambda c: (c.t / "registrations/R24_multiwoz_random_negatives.md").write_text("changed\n")),
    ("on R24's path, unchanged: scripts/knob_run.py", lambda c: (c.t / "scripts/knob_run.py").write_text("# changed\n")),
    ("other changes since the freeze", lambda c: (c.t / "scripts/new_tool.py").write_text("# new\n")),
    ("other changes since the freeze: ['scripts/r24_check.py']", lambda c: open(c.t / "scripts/r24_check.py", "a").write("# loosened\n")),
    ("logs/jobs_R24.txt is e85_pick.py's r24_jobs at s*, line for line", job_s12),
    ("s* from logs/e85_pick_final.out", lambda c: (c.t / "logs/e85_pick_final.out").write_text("s*: refined 16, C4 8\n")),
    ("dev row e85_multiwoz_a3B_c16000_s16_nrand seed 1: exactly one",
     lambda c: (c.t / "results/explore/abcd.jsonl").write_text((c.t / "results/explore/abcd.jsonl").read_text() * 2)),
    ("seed files are exactly", lambda c: shutil.copy(c.js("tm_mem_c4_nrand", 5), c.js("tm_mem_c4_nrand", 6))),
    ("examples_test.json.gz written after the freeze", lambda c: old(c.t / "data/multiwoz/examples_test.json.gz", c)),
    ("examples_test uses train's label list", lambda c: gzip.open(c.t / "data/multiwoz/examples_test.json.gz", "wt").write('{"buttons": ["b", "a"]}')),
    ("the bge-large dev-stage file has its frozen hash",
     lambda c: np.savez(c.t / "data/emb/multiwoz__bge-large.devstage.npz", provenance=np.array("other"), train__cust=np.ones((3, 2)), dev__cust=np.zeros((2, 2)))),
    ("train/dev arrays equal the dev-stage file's",
     lambda c: np.savez(c.t / "data/emb/multiwoz__bge-large.npz", train__cust=np.zeros((3, 2)), dev__cust=np.zeros((2, 2)))),
    ("R24mwoz/tm_mem_ref_nrand/seed2: encoder none, parts, random negatives",
     lambda c: c.edit("tm_mem_ref_nrand", 2, lambda j: j["cfg"].update(neg_sampling="hardest"))),
    ("R24mwoz/tm_mem_c4_nrand/seed2: init half", lambda c: c.edit("tm_mem_c4_nrand", 2, lambda j: j.update(init="zeros"))),
    ("R24mwoz/tm_mem_ref_nrand/seed3: the arm's table",
     lambda c: c.edit("tm_mem_ref_nrand", 3, lambda j: j["static_info"]["xcust"].update(table_file_sha256="0" * 64))),
    ("R24mwoz/tm_mem_ref_nrand/seed2: cfg equals", lambda c: c.edit("tm_mem_ref_nrand", 2, lambda j: j["cfg"].update(s=8.0))),
    ("R24mwoz/tm_mem_c4_nrand/seed4: written", lambda c: c.edit("tm_mem_c4_nrand", 4, lambda j: j.update(time=stamp(c.freeze - 60)))),
    ("R24mwoz/tm_mem_bge_nrand/seed2: train/dev bits equal", lambda c: c.edit("tm_mem_bge_nrand", 2, lambda j: j["bits_sha256"].update(train="0" * 64))),
    ("R24mwoz/tm_mem_c4_nrand/seed1: dev accuracy", lambda c: c.edit("tm_mem_c4_nrand", 1, lambda j: j.update(dev_acc=ACC["c4"] + 1e-6))),
    ("R24mwoz/tm_mem_bge_nrand/seed3: npz gives the json's test accuracy",
     lambda c: c.edit("tm_mem_bge_nrand", 3, lambda j: j.update(test_acc=j["test_acc"] + 0.001))),
    ("R24mwoz/tm_mem_c4_nrand/seed2: test rows equal the other arms'", lambda c: npz_set(c, "tm_mem_c4_nrand", 2, "y")),
    ("R24mwoz/tm_mem_bge_nrand/seed4: test rows equal the other arms'", lambda c: npz_set(c, "tm_mem_bge_nrand", 4, "k")),
    ("tm_mem_ref_nrand: one test-bit hash over seeds", lambda c: c.edit("tm_mem_ref_nrand", 5, lambda j: j["bits_sha256"].update(test="1" * 64))),
    ("lr_mem_c4: bits of its TM arm; dev accuracy and C", lambda c: c.edit("lr_mem_c4", 0, lambda j: j.update(C=0.003))),
    ("lr_mem_bge: the same test rows", lambda c: npz_set(c, "lr_mem_bge", 0, "w_roberta")),
]


def main() -> None:
    c = Case()
    p = c.run()
    assert p.returncode == 0 and p.stdout.strip().endswith("all checks passed"), p.stdout + p.stderr
    assert "s* from logs/e85_pick_final.out: refined s 16, bge-large s 8, C4 s 8" in p.stdout, p.stdout
    shutil.rmtree(c.t)
    print("ok full mode passes on consistent files (refined at s* 16 against its e85b dev row)")
    for why, change in REFUSALS:
        c = Case()
        change(c)
        p = c.run()
        last = (p.stdout.strip().splitlines() or [""])[-1]
        assert p.returncode == 1 and last.startswith("FAIL ") and why in last, (why, p.stdout[-600:], p.stderr[-600:])
        shutil.rmtree(c.t)
    c = Case()
    ex = c.t / "data/multiwoz/examples_test.json.gz"
    with gzip.open(ex, "rt") as f:
        d = json.load(f)
    d["rows"].append({"convo_id": 5000})
    with gzip.open(ex, "wt") as f:
        json.dump(d, f)
    p = c.run()
    assert p.returncode == 1 and "FAIL 2000 test turns in 1000 dialogues, as in examples_test.json.gz (2001 in 1001)" in p.stdout, p.stdout[-300:]
    shutil.rmtree(c.t)
    c = Case(dialogues=999)
    p = c.run()
    assert p.returncode == 0 and "ok  1998 test turns in 999 dialogues, as in examples_test.json.gz (1998 in 999)" in p.stdout, p.stdout[-300:]
    shutil.rmtree(c.t)
    print(f"ok {len(REFUSALS) + 1} refusals, each on its own check's line (the last: a test turn missing from the arms); 999 dialogues pass when the examples hold 999")
    c = Case(acc={"ref": 0.36, "c4": 0.33, "bge": 0.345})
    p = c.run("--gate")
    assert p.returncode == 0 and "G-s* C4: e84_multiwoz_C4_c16000_s8_nrand 0.3300 minus e84_multiwoz_bge_c16000_s8_nrand 0.3450: -1.50" in p.stdout \
        and "missed; its arm runs with no rule" in p.stdout and "G-s* refined: e85_multiwoz_a3B_c16000_s16_nrand 0.3600" in p.stdout \
        and "+1.50 (needs >= -1.0): holds" in p.stdout and p.stdout.strip().endswith("tm_mem_ref_nrand keep their rules"), p.stdout + p.stderr
    shutil.rmtree(c.t)
    c = Case(acc={"ref": 0.33, "c4": 0.33, "bge": 0.345})
    p = c.run("--gate")
    assert p.returncode == 3 and "both agents missed" in p.stdout, p.stdout + p.stderr
    shutil.rmtree(c.t)
    c = Case(acc={"ref": 0.34, "c4": 0.34, "bge": 0.35})
    p = c.run("--gate")
    assert p.returncode == 0 and p.stdout.count("-1.00 (needs >= -1.0): holds") == 2, p.stdout + p.stderr
    shutil.rmtree(c.t)
    print("ok --gate: one agent missed keeps exit 0 and says so; both missed exit 3; exactly -1.0 holds")
    print("PASSED")


if __name__ == "__main__":
    main()
