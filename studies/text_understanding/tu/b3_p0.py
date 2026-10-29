#!/usr/bin/env python
"""Step 3b, the CPU pilot P0 (tu/PLAN_3b.md, "Tests before the GPU job"): is the TM student near the linear student on
C4 held-out bits? Dev only: C4 sentences and their teacher vectors, no intent text at all.

    python -I -B b3_p0.py --choice tu/b3/choice.json --codes tu/b3/codes.npz --out-dir tu/b3/p0 [--n-train 100000
        --n-held 10000 --s 4,8,16]

Steps (each one is a tested CLI run as a subprocess; a step whose output exists is skipped, so a stopped pilot can be
started again; nothing is overwritten):
1. b3_c4_feats.py fit on all of c4_sent_train_f.txt (CPU concepts, K 512);
2. b3_c4_feats.py encode: the first --n-train training lines and the first --n-held held-out lines;
3. the chosen code (choice.json "pick") on the matching rows of T_train and T_held -> train.npz and held.npz (X and all
   768 code bits as Y), then train64.npz and held64.npz with the --bits code bits fixed by numpy
   default_rng(20261031).choice(768, --bits, replace=False), sorted. Reason (measured 2026-10-21 07:48Z before P0 ran):
   one feedback call takes 1.2 s on 2 CPU threads at 1,000 clauses, whatever the number of outputs, and each call
   trains one output per row; with all 768 outputs a CPU pilot would give each output about 1,600 row-feedbacks, far
   fewer than the planted check needed (about 16,000). With 64 outputs each gets about 18,750. The GPU job (2M rows,
   all 768 outputs) gives each output more than that, so P0 on 64 bits is the conservative reading;
4. b3_student.py train for each s on the 64-bit files: 1,000 clauses, T 1,000, R 4, batch 1,024, 3 epochs, seed 1, CPU;
5. b3_twins.py (ridge and counting) on the same 64-bit files;
6. the GO rule: GO when the best TM student's held-out mean balanced bit accuracy is at least the linear twin's minus
   2.0 points; written to p0_verdict.json with every input's sha256. Fidelity (CRISP's measure: the held-out bitwise
   agreement, "bit_acc") is reported beside it for every student.
--pool-check (descriptive, after the verdict; the note of 07:32Z): on 32 code bits fixed by numpy
default_rng(20261029).choice(768, 32, replace=False), sorted, at the verdict's best s and matched total clauses: one
shared pool (1,024 clauses, T 1,024, R 4) against 32 CRISP-style per-bit TMs (32 clauses and T 32 each, R 1), the same
rows and epochs; mean balanced accuracy and fidelity over the 32 bits -> p0_pool_check.json.
--fit-limit, --sent-train, --sent-held, --t-train and --t-held exist for the smoke test only (the pilot uses all of
the training sentences for the concepts and 3a's files).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
A3 = HERE / "a3"
PY = sys.executable
GO_MARGIN = 2.0


def sha16(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def run(cmd: list[str], log: Path) -> None:
    with open(log, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} run: {' '.join(cmd)}\n")
        f.flush()
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} exit {p.returncode}\n")
    if p.returncode != 0:
        raise SystemExit(f"step failed (exit {p.returncode}): {' '.join(cmd)}; see {log}")


def targets(codes: Path, pick: str, T: Path, n: int) -> tuple[np.ndarray, int]:
    sys.path.insert(0, str(HERE))
    import b3_targets as BT

    c = dict(np.load(codes))
    bits = BT.apply_code(c, pick, np.asarray(np.load(T, mmap_mode="r")[:n], np.float32))
    return np.packbits(bits, axis=1), bits.shape[1]


def pool_check(a, d: Path) -> int:
    out = d / "p0_pool_check.json"
    if out.exists():
        print(f"refusing to overwrite {out}", file=sys.stderr)
        return 2
    v = json.loads((d / "p0_verdict.json").read_text())
    s = v["best_s"]
    log = d / "p0.log"
    bits = np.sort(np.random.default_rng(20261029).choice(768, 32, replace=False))
    for k in ("train", "held"):
        z = np.load(d / f"{k}.npz")
        Y = np.unpackbits(z["Y"], axis=1)[:, : int(z["n_outputs"])][:, bits]
        p32 = d / f"{k}32.npz"
        if not p32.exists():
            with open(p32, "xb") as fo:
                np.savez(fo, X=z["X"], Y=np.packbits(Y, axis=1), n_features=z["n_features"], n_outputs=np.asarray(32))
        for j in range(32):
            p1 = d / "perbit" / f"{k}_b{j:02d}.npz"
            if not p1.exists():
                p1.parent.mkdir(exist_ok=True)
                with open(p1, "xb") as fo:
                    np.savez(fo, X=z["X"], Y=np.packbits(Y[:, j : j + 1], axis=1), n_features=z["n_features"], n_outputs=np.asarray(1))
    st = [PY, "-I", "-B", str(HERE / "b3_student.py"), "train", "--epochs", str(a.epochs), "--batch", "1024", "--seed", "1",
          "--device", "cpu", "--s", s]
    if not (d / "pool_shared32.json").exists():
        run(st + ["--train", str(d / "train32.npz"), "--held", str(d / "held32.npz"), "--out-dir", str(d), "--tag", "pool_shared32",
                  "--clauses", "1024", "--T", "1024", "--R", "4"], log)
    per = []
    for j in range(32):
        tag = f"pool_bit{j:02d}"
        if not (d / "perbit" / f"{tag}.json").exists():
            run(st + ["--train", str(d / "perbit" / f"train_b{j:02d}.npz"), "--held", str(d / "perbit" / f"held_b{j:02d}.npz"),
                      "--out-dir", str(d / "perbit"), "--tag", tag, "--clauses", "32", "--T", "32", "--R", "1"], log)
        per.append(json.loads((d / "perbit" / f"{tag}.json").read_text())["curve"][-1])
    sh = json.loads((d / "pool_shared32.json").read_text())
    res = {"bits": bits.tolist(), "s": s, "epochs": a.epochs, "shared": {"clauses": 1024, "R": 4, "held": sh["curve"][-1],
           "seconds": sh["seconds"]},
           "per_bit": {"clauses_each": 32, "R": 1, "mean_balanced_acc": round(float(np.mean([r["balanced_acc"] for r in per])), 3),
                       "mean_bit_acc": round(float(np.mean([r["bit_acc"] for r in per])), 3),
                       "constant_bits": int(sum(r["constant_outputs"] for r in per)),
                       "seconds": round(sum(json.loads((d / "perbit" / f"pool_bit{j:02d}.json").read_text())["seconds"] for j in range(32)), 1)},
           "note": "descriptive; matched total clauses (1,024 against 32 x 32); the per-bit TMs give each bit feedback on every row",
           "b3_p0.py": sha16(Path(__file__).resolve()), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(out, "x") as fo:
        fo.write(json.dumps(res, indent=1) + "\n")
    print(json.dumps({"shared": res["shared"]["held"], "per_bit_balanced": res["per_bit"]["mean_balanced_acc"],
                      "per_bit_fidelity": res["per_bit"]["mean_bit_acc"]}))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--choice", type=Path, required=True)
    ap.add_argument("--codes", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-train", type=int, default=100_000)
    ap.add_argument("--n-held", type=int, default=10_000)
    ap.add_argument("--s", default="4,8,16")
    ap.add_argument("--clauses", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bits", type=int, default=64)
    ap.add_argument("--pool-check", action="store_true")
    ap.add_argument("--fit-limit", type=int, default=0)
    ap.add_argument("--sent-train", type=Path, default=A3 / "c4_sent_train_f.txt")
    ap.add_argument("--sent-held", type=Path, default=A3 / "c4_sent_held_f.txt")
    ap.add_argument("--t-train", type=Path, default=A3 / "T_train.f16.npy")
    ap.add_argument("--t-held", type=Path, default=A3 / "T_held.f16.npy")
    a = ap.parse_args()
    d = a.out_dir
    d.mkdir(parents=True, exist_ok=True)
    verdict = d / "p0_verdict.json"
    if verdict.exists() and not a.pool_check:
        print(f"refusing to overwrite {verdict}", file=sys.stderr)
        return 2
    if a.pool_check:
        return pool_check(a, d)
    log = d / "p0.log"
    pick = json.loads(a.choice.read_text())["pick"]
    cf = [PY, "-I", "-B", str(HERE / "b3_c4_feats.py")]
    conc = d / "c4_concepts_cpu.npz"
    if not conc.exists():
        run(cf + ["fit", "--sentences", str(a.sent_train), "--out", str(conc), "--device", "cpu", "--limit", str(a.fit_limit)], log)
    xs = {"train": (d / "x_train.npz", a.sent_train, a.n_train), "held": (d / "x_held.npz", a.sent_held, a.n_held)}
    for x, src, n in xs.values():
        if not x.exists():
            run(cf + ["encode", "--concepts", str(conc), "--sentences", str(src), "--limit", str(n), "--out", str(x),
                      "--device", "cpu"], log)
    data = {"train": (d / "train.npz", a.t_train), "held": (d / "held.npz", a.t_held)}
    for k, (out, T) in data.items():
        if not out.exists():
            z = np.load(xs[k][0])
            Yp, J = targets(a.codes, pick, T, xs[k][2])
            assert z["X"].shape[0] == Yp.shape[0], (k, z["X"].shape, Yp.shape)
            with open(out, "xb") as fo:
                np.savez(fo, X=z["X"], Y=Yp, n_features=z["n_features"], n_outputs=np.asarray(J),
                         code=np.asarray(pick), x_sha16=np.asarray(sha16(xs[k][0])), T_sha16=np.asarray(sha16(T)))
    sel = np.sort(np.random.default_rng(20261031).choice(768, a.bits, replace=False))
    for k in ("train", "held"):
        sub = d / f"{k}{a.bits}.npz"
        if not sub.exists():
            z = np.load(data[k][0])
            Y = np.unpackbits(z["Y"], axis=1)[:, : int(z["n_outputs"])][:, sel]
            with open(sub, "xb") as fo:
                np.savez(fo, X=z["X"], Y=np.packbits(Y, axis=1), n_features=z["n_features"], n_outputs=np.asarray(a.bits),
                         bits=sel, code=z["code"])
    data = {k: (d / f"{k}{a.bits}.npz", v[1]) for k, v in data.items()}
    st = [PY, "-I", "-B", str(HERE / "b3_student.py"), "train", "--train", str(data["train"][0]), "--held", str(data["held"][0]),
          "--out-dir", str(d), "--clauses", str(a.clauses), "--T", str(a.clauses), "--R", "4", "--epochs", str(a.epochs),
          "--batch", "1024", "--seed", "1", "--device", "cpu"]
    tms = {}
    for s in a.s.split(","):
        tag = f"tm_s{s}"
        if not (d / f"{tag}.json").exists():
            run(st + ["--tag", tag, "--s", s], log)
        tms[s] = json.loads((d / f"{tag}.json").read_text())
    if not (d / "twins.twins.json").exists():
        run([PY, "-I", "-B", str(HERE / "b3_twins.py"), "--train", str(data["train"][0]), "--held", str(data["held"][0]),
             "--out-dir", str(d), "--tag", "twins"], log)
    tw = json.loads((d / "twins.twins.json").read_text())
    best_s = max(tms, key=lambda s: tms[s]["curve"][-1]["balanced_acc"])
    tm_best = tms[best_s]["curve"][-1]["balanced_acc"]
    lin = tw["linear"]["held"]["balanced_acc"]
    res = {"code": pick, "bits": sel.tolist(), "n_bits": a.bits, "n_train": a.n_train, "n_held": a.n_held, "clauses": a.clauses,
           "epochs": a.epochs,
           "tm_by_s": {s: tms[s]["curve"][-1] for s in tms}, "tm_seconds_by_s": {s: tms[s]["seconds"] for s in tms},
           "best_s": best_s, "tm_best_balanced_acc": tm_best, "tm_best_fidelity": tms[best_s]["curve"][-1]["bit_acc"],
           "linear": tw["linear"]["held"], "linear_lam": tw["linear"]["lam_factor"],
           "counting": tw["counting"]["held"], "rule": f"GO when the best TM >= linear - {GO_MARGIN} (mean balanced bit accuracy, C4 held-out)",
           "difference_tm_minus_linear": round(tm_best - lin, 3), "go": bool(tm_best >= lin - GO_MARGIN - 1e-9),
           "files_sha16": {p.name: sha16(p) for p in sorted(d.glob("*.npz")) + sorted(d.glob("*.json")) if p != verdict},
           "inputs_sha16": {str(a.choice): sha16(a.choice), str(a.codes): sha16(a.codes)},
           "b3_p0.py": sha16(Path(__file__).resolve()), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(verdict, "x") as fo:
        fo.write(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("code", "best_s", "tm_best_balanced_acc", "difference_tm_minus_linear", "go")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
