#!/usr/bin/env python
"""Tests for M1's code (tu/PLAN_M1.md). A plain script, no pytest; CPU only, no GPU. It uses no test text and no test
label: every run on real data uses the stand-in datasets of m1_standin.py, which replace the test split with a copy of
val as soon as dl.data.load returns.

    OMP_NUM_THREADS=2 nice -n 19 $PY -I -B tu/tests/test_m1.py [--feats]

- m1_verdict.py on synthetic rows and predictions:
  - the point estimates, a bar met exactly at its boundary, "strong" from the bootstrap's lower end;
  - the per-seed differences, the seed SD, the accuracy without the step-0 matches;
  - the headline; identical output on a rerun (the fixed generator);
  - the refusals: a missing seed, a smoke row, s 16, a duplicate LR row, a row that disagrees with its predictions,
    an overwrite.
- m1_overlap.keys / overlap on synthetic texts. m1_overlap.py with tiny C4 and kept files, planted copies, the
  stand-in dataset and --skip-check-sentences.
- m1_check_cache.py on a synthetic cache: test := val, and the refusals.
- With --feats (a few minutes):
  - m1_feats.py on the CPU for HWU64 (the stand-in with 1,500 training rows): a3B with a 5-row kept file, and potion;
    one k-means per table, test rows equal val rows, the centroids saved;
  - m1_tm.py --smoke and m1_lr.py on the CPU: one row each, "test" accuracy = val accuracy, the predictions saved,
    and a second run with the same key refused (one test read per run).
  - The TM's fit calls torch.cuda.synchronize() unconditionally, so m1_tm.py runs through a wrapper that makes that
    call a no-op (this test only).
Each run has its own folder in tu/tests/runs/<UTC stamp>_m1/.
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

TU = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TU))
sys.path.insert(0, str(TU.parent / "w1"))
sys.path.insert(0, f"{_REPO_ROOT}/studies/decision_layer/src")
PY = sys.executable
RUNS = TU / "tests" / "runs" / (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_m1")
STANDIN = str(TU / "m1_standin.py")
results = {"pass": 0, "fail": 0}


def check(name: str, ok: bool, detail: str = "") -> None:
    results["pass" if ok else "fail"] += 1
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f": {detail}"))


def run(args: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run([PY, "-I", "-B", *args], capture_output=True, text=True, timeout=timeout, check=False)


RUNS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- m1_verdict.py on synthetic rows

N = {"clinc150": 1000, "banking77": 800, "hwu64": 600}
# correct predictions per configuration, as counts out of N (seed 7, 8, 9 for TM)
TMC = {"clinc150": {("C4", 1): (899, 900, 901), ("a3B", 1): (920,) * 3, ("potion", 1): (925,) * 3,
                    ("a3B_3c", 1): (930,) * 3, ("C4_3c", 1): (910,) * 3, ("a3B_twin", 1): (925,) * 3, ("a3B", 5): (925,) * 3},
       "banking77": {("C4", 1): (704,) * 3, ("a3B", 1): (706,) * 3, ("potion", 1): (704,) * 3, ("a3B_3c", 1): (710,) * 3,
                     ("C4_3c", 1): (705,) * 3, ("a3B_twin", 1): (708,) * 3, ("a3B", 5): (707,) * 3},
       "hwu64": {("C4", 1): (540,) * 3, ("a3B", 1): (546,) * 3, ("potion", 1): (552,) * 3, ("a3B_3c", 1): (550,) * 3,
                 ("C4_3c", 1): (545,) * 3, ("a3B_twin", 1): (548,) * 3, ("a3B", 5): (547,) * 3}}
LRC = {"clinc150": {"C4": 900, "a3B": 930, "potion": 932, "a3B_3c": 940, "C4_3c": 915, "a3B_twin": 934},
       "banking77": {"C4": 696, "a3B": 720, "potion": 712, "a3B_3c": 722, "C4_3c": 700, "a3B_twin": 704},
       "hwu64": {"C4": 540, "a3B": 546, "potion": 552, "a3B_3c": 552, "C4_3c": 541, "a3B_twin": 552}}


def make_synth(d0: Path) -> None:
    (d0 / "preds").mkdir(parents=True)
    tm, lr = [], []
    for d, n in N.items():
        for (arm, rep), ks in TMC[d].items():
            for sd, k in zip((7, 8, 9), ks):
                p = d0 / "preds" / f"m1_{arm}_{d}_r{rep}_s{sd}.npy"
                np.save(p, np.r_[np.zeros(k), np.ones(n - k)].astype(np.int16))
                tm.append({"tag": "m1", "dataset": d, "arm": arm, "repeat": rep, "seed": sd, "s": 8.0, "smoke": False,
                           "cfg": {"n_clauses": 4000, "T": 4000, "seed": sd}, "test_acc": round(k / n, 4),
                           "val_acc": round(k / n + 0.01, 4), "preds": str(p)})
        for arm, k in LRC[d].items():
            p = d0 / "preds" / f"lr_{arm}_{d}.npy"
            np.save(p, np.r_[np.zeros(k), np.ones(n - k)].astype(np.int16))
            lr.append({"dataset": d, "name": arm, "lr_test": round(k / n, 4), "lr_val": round(k / n, 4), "C": 1.0, "preds": str(p)})
    (d0 / "results_m1_tm.jsonl").write_text("".join(json.dumps(r) + "\n" for r in tm))
    (d0 / "results_m1_lr.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lr))
    (d0 / "labels.py").write_text(
        "import numpy as np\n\nclass D:\n    pass\n\n\ndef load(name):\n    d = D()\n"
        f"    d.labels = {{'test': np.zeros({{'clinc150': {N['clinc150']}, 'banking77': {N['banking77']}, 'hwu64': {N['hwu64']}}}[name], np.int64)}}\n"
        "    return d\n")
    ov = {"sets": {d: {"matched_test_idx": [0, 1, 2, 3, 4] if d == "clinc150" else []} for d in N}}
    (d0 / "overlap.json").write_text(json.dumps(ov))


def verdict(d0: Path) -> subprocess.CompletedProcess:
    return run([str(TU / "m1_verdict.py"), "--m1", str(d0), "--overlap", str(d0 / "overlap.json"), "--labels",
                f"{d0 / 'labels.py'}:load"])


v1 = RUNS / "verdict1"
make_synth(v1)
p = verdict(v1)
check("verdict: exit 0", p.returncode == 0, p.stderr[-800:])
if p.returncode == 0:
    J = json.loads((v1 / "verdict_m1.json").read_text())
    c = J["sets"]["clinc150"]["comparisons"]
    check("verdict: point estimates from the predictions (H1 TM +2.00, LR +3.00 on clinc150)",
          abs(c["H1_tm"]["point"] - 2.0) < 1e-6 and abs(c["H1_lr"]["point"] - 3.0) < 1e-6, str(c["H1_tm"]))
    check("verdict: per-seed differences and seed SE (2.1, 2.0, 1.9)",
          c["H1_tm"]["per_seed"] == [2.1, 2.0, 1.9] and abs(c["H1_tm"]["seed_se"] - 0.0577) < 1e-3, str(c["H1_tm"]))
    check("verdict: the C4 arm's seed SD is 0.1", abs(J["sets"]["clinc150"]["tm"]["C4@1"]["test_seed_sd"] - 0.1) < 1e-6)
    h = {d: J["sets"][d]["hypotheses"] for d in N}
    check("verdict: a bar met exactly at its boundary counts (H2 TM -0.50 on clinc150), and is not strong",
          abs(c["H2_tm"]["point"] + 0.5) < 1e-6 and h["clinc150"]["H2"] == {"met": True, "strong": False}, str(h["clinc150"]))
    check("verdict: H1 strong on clinc150 (+2.0 and +3.0 at n 1,000), not on hwu64 (+1.0 at n 600)",
          h["clinc150"]["H1"] == {"met": True, "strong": True} and h["hwu64"]["H1"] == {"met": True, "strong": False},
          str(h))
    check("verdict: the interval holds its point and is about 2 x 1.96 SEs wide (H1 TM clinc150)",
          c["H1_tm"]["ci95"][0] < 2.0 < c["H1_tm"]["ci95"][1] and 1.2 < c["H1_tm"]["ci95"][1] - c["H1_tm"]["ci95"][0] < 2.3,
          str(c["H1_tm"]["ci95"]))
    check("verdict: misses where designed (H1 banking77 TM +0.25; H3a banking77 +0.25; H3b hwu64 0.0; H2 hwu64 TM -1.0)",
          not h["banking77"]["H1"]["met"] and not h["banking77"]["H3a"]["met"] and not h["hwu64"]["H3b"]["met"]
          and not h["hwu64"]["H2"]["met"], str(h))
    check("verdict: H3b strong on banking77 (+2.25 at n 800)", h["banking77"]["H3b"] == {"met": True, "strong": True}, str(h["banking77"]))
    head = (v1 / "verdict_m1.txt").read_text().splitlines()[0]
    check("verdict: the headline", head == "VERDICT M1 (test): H1 MET (2 of 3, strong on 1); H2 MET (2 of 3, strong on 1); "
          "H3a MET (2 of 3); H3b MET (2 of 3, strong on 1)", head)
    w = J["sets"]["clinc150"]["without_matched_test_texts"]
    check("verdict: accuracy without the 5 matched texts (a3B@1: 915 / 995)", w["n_dropped"] == 5
          and abs(w["tm"]["a3B@1"] - round(100 * 915 / 995, 4)) < 1e-6, str(w))
    txt = (v1 / "verdict_m1.txt").read_text()
    check("verdict: the multiplicity and interval-limit lines are written",
          "separate claims with no correction" in txt and "leave out the variance between training seeds" in txt)
    p2 = verdict(v1)
    check("verdict: refuses to overwrite", p2.returncode == 2, p2.stderr[-300:])
    v2 = RUNS / "verdict2"
    make_synth(v2)
    verdict(v2)
    J2 = json.loads((v2 / "verdict_m1.json").read_text()) if (v2 / "verdict_m1.json").exists() else {}
    check("verdict: identical output on a rerun (fixed bootstrap generator)", J2.get("sets") == J["sets"])


def broken(name: str, edit) -> None:
    d0 = RUNS / f"verdict_{name}"
    make_synth(d0)
    tm = [json.loads(x) for x in (d0 / "results_m1_tm.jsonl").read_text().splitlines()]
    lr = [json.loads(x) for x in (d0 / "results_m1_lr.jsonl").read_text().splitlines()]
    tm, lr, want = edit(tm, lr)
    (d0 / "results_m1_tm.jsonl").write_text("".join(json.dumps(r) + "\n" for r in tm))
    (d0 / "results_m1_lr.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lr))
    p = verdict(d0)
    check(f"verdict refuses: {name} (exit {want})", p.returncode == want and not (d0 / "verdict_m1.json").exists(),
          f"exit {p.returncode} {p.stderr[-300:]}")


broken("a missing seed", lambda tm, lr: (tm[1:], lr, 3))
broken("a smoke row", lambda tm, lr: (tm + [{**tm[0], "smoke": True}], lr, 3))
broken("s 16", lambda tm, lr: ([{**tm[0], "s": 16.0}] + tm[1:], lr, 3))
broken("a duplicate LR row", lambda tm, lr: (tm, lr + [lr[0]], 3))
broken("a row that disagrees with its predictions", lambda tm, lr: ([{**tm[0], "test_acc": 0.5}] + tm[1:], lr, 4))

# ---------------------------------------------------------------- m1_overlap
import m1_overlap as O

live = {"turn", "lights", "off", "kitchen"}
norm = lambda t: " ".join(t.lower().split())
tok = lambda t: t.lower().split()
e, o, r = O.keys("Turn the lights OFF", live, norm, tok)
check("overlap: keys (exact, order-free, reader)", e == "turn the lights off" and o == ("lights", "off", "the", "turn")
      and r == frozenset({"turn", "lights", "off"}), str((e, o, r)))
res = O.overlap(["turn the lights off", "off lights the turn", "zz qq"],
                {"x:exact": {"turn the lights off"}, "x:order_free": {("lights", "off", "the", "turn")},
                 "x:reader": {frozenset({"turn", "lights", "off"})}}, live, norm, tok)
check("overlap: exact hits 1, order-free 2, reader 2; an empty reader key never matches",
      res["counts"] == {"x:exact": 1, "x:order_free": 2, "x:reader": 2} and res["idx"]["x:reader"] == [0, 1], str(res["counts"]))

c4u = RUNS / "c4_unit.txt"
c4u.write_text("Turn the lights OFF\noff lights the turn\nnothing here\n")
hh, nn = O.c4_hits([c4u], {"s": ["turn the lights off", "lights off", "the turn off lights"]}, norm)
check("overlap: c4_hits reads every line once and finds the exact and the order-free copies",
      nn == 3 and hh["s"]["c4:exact"] == {0} and hh["s"]["c4:order_free"] == {0, 2}, str((nn, hh)))

sys.path.insert(0, str(TU))
import m1_standin as S

hv = S.load("hwu64").texts["val"]
k0 = next(i for i, t in enumerate(hv) if len(t.split()) >= 4 and hv.count(t) == 1)
k1 = next(i for i, t in enumerate(hv) if len(t.split()) >= 4 and hv.count(t) == 1 and i > k0 + 10)
t0, t1 = hv[k0], hv[k1]
c4f = RUNS / "tiny_c4.txt"
c4f.write_text(f"{t0}\n{' '.join(reversed(t1.split()))}\nan unrelated sentence about rivers and mountains\n")
kf = RUNS / "tiny_kept.jsonl"
kf.write_text(json.dumps({"set": "hwu64", "text": t1 + " zzqxw"}) + "\n" + json.dumps({"set": "banking77", "text": t0}) + "\n")
ovp = RUNS / "overlap_test.json"
p = run([str(TU / "m1_overlap.py"), "--out", str(ovp), "--c4", str(c4f), "--kept", str(kf), "--load", f"{STANDIN}:load",
         "--skip-check-sentences"], timeout=900)
check("overlap CLI: exit 0", p.returncode == 0, p.stderr[-800:])
if p.returncode == 0:
    OJ = json.loads(ovp.read_text())
    hw = OJ["sets"]["hwu64"]
    check("overlap CLI: the planted exact copy (C4) is found", hw["counts"]["c4:exact"] >= 1 and k0 in hw["matched_test_idx"], str(hw["counts"]))
    check("overlap CLI: the planted reordered copy (C4) is found by the order-free key, not the exact key",
          hw["counts"]["c4:order_free"] >= 2 and k1 in hw["idx_by_key"]["c4:order_free"]
          and k1 not in hw["idx_by_key"]["c4:exact"] and k1 in hw["matched_test_idx"], str(hw["counts"]))
    check("overlap CLI: the kept row with an out-of-table word matches by the reader key only",
          k1 in hw["idx_by_key"]["kept:reader"] and k1 not in hw["idx_by_key"]["kept:order_free"]
          and hw["counts"]["kept:exact"] == 0, str(hw["counts"]))
    check("overlap CLI: a kept row of another set does not count for hwu64; n_test = the stand-in's val",
          hw["n_test"] == len(hv) and OJ["sets"]["banking77"]["counts"]["kept:exact"] == 0, str(OJ["sets"]["banking77"]["counts"]))
    check("overlap CLI: --skip-check-sentences reads no evaluation text", OJ["a3_teacher_check_sentences_in_eval"].startswith("not checked"))
    p2 = run([str(TU / "m1_overlap.py"), "--out", str(ovp), "--c4", str(c4f), "--kept", str(kf), "--load", f"{STANDIN}:load",
              "--skip-check-sentences"])
    check("overlap CLI: refuses to overwrite", p2.returncode == 2, p2.stderr[-300:])

# ---------------------------------------------------------------- m1_check_cache.py
src = RUNS / "cc_src"
src.mkdir()
np.savez(src / "clinc150__p32tok_K512.npz", train__exists=np.ones((4, 3), np.float32), val__exists=np.arange(6, dtype=np.float32).reshape(2, 3),
         y_train=np.arange(4), y_val=np.array([5, 6]))
(src / "clinc150__p32tok_K512.meta.json").write_text(json.dumps({"table": "a3B_d768", "arm": "orig"}))
cco = RUNS / "cc_out"
p = run([str(TU / "m1_check_cache.py"), "--src", str(src), "--out", str(cco)])
check("check cache: exit 0", p.returncode == 0, p.stderr[-500:])
if p.returncode == 0:
    zc = np.load(cco / "clinc150__p32tok_K512.npz")
    mc = json.loads((cco / "clinc150__p32tok_K512.meta.json").read_text())
    check("check cache: test := val, y_test := y_val, the meta says so",
          np.array_equal(zc["test__exists"], zc["val__exists"]) and list(zc["y_test"]) == [5, 6]
          and mc["m1_code_check"]["test_is_a_copy_of_val"] and mc["table"] == "a3B_d768")
    p2 = run([str(TU / "m1_check_cache.py"), "--src", str(src), "--out", str(cco)])
    check("check cache: refuses to overwrite", p2.returncode == 2, p2.stderr[-300:])
    p3 = run([str(TU / "m1_check_cache.py"), "--src", str(cco), "--out", str(RUNS / "cc_out2")])
    check("check cache: refuses a source that has test rows", p3.returncode != 0 and "has test rows" in p3.stderr, p3.stderr[-300:])

# ---------------------------------------------------------------- --feats: the real code paths on the CPU
if "--feats" in sys.argv:
    ds = S.load_small("hwu64")
    kept = RUNS / "tiny_kept_hwu.jsonl"
    rows = [{"set": "hwu64", "index": i, "y": int(ds.labels["train"][i]), "label": ds.intents[int(ds.labels["train"][i])],
             "source": ds.texts["train"][i], "text": ds.texts["train"][i] + " please"} for i in range(5)]
    kept.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = RUNS / "feats"
    t_0 = time.time()
    p = run([str(TU / "m1_feats.py"), "--dataset", "hwu64", "--table", "wordtab:" + str(TU / "a3" / "tables" / "a3B_d768.npz"),
             "--arms", f"a3B:none,a3B_3c:{kept}", "--out-root", str(out), "--device", "cpu", "--load", f"{STANDIN}:load_small"],
            timeout=1800)
    check(f"feats a3B: exit 0 ({time.time() - t_0:.0f} s)", p.returncode == 0, p.stderr[-800:])
    if p.returncode == 0:
        za, zb = np.load(out / "a3B" / "hwu64__p32tok_K512.npz"), np.load(out / "a3B_3c" / "hwu64__p32tok_K512.npz")
        ma = json.loads((out / "a3B" / "hwu64__p32tok_K512.meta.json").read_text())
        mb = json.loads((out / "a3B_3c" / "hwu64__p32tok_K512.meta.json").read_text())
        cen = np.load(out / "a3B" / "hwu64__p32tok_K512.centroids.npy")
        check("feats a3B: both arms share one k-means; the centroids are saved (512 x 768)",
              ma["kmeans_run_id"] == mb["kmeans_run_id"] and cen.shape == (512, 768), str(cen.shape))
        check("feats a3B: the stand-in's test rows equal its val rows; y_test = y_val; 'test' in the splits",
              all(np.array_equal(za[f"test__{k}"], za[f"val__{k}"]) for k in ("mean", "exists"))
              and np.array_equal(za["y_test"], za["y_val"]) and "test" in ma["splits"])
        check("feats a3B: 1,500 originals plus 5 kept rows with their labels; the originals equal the a3B arm's",
              zb["y_train"].shape[0] == 1505 and list(zb["y_train"][1500:]) == [r["y"] for r in rows]
              and np.array_equal(za["train__exists"], zb["train__exists"][:1500]))
    t_0 = time.time()
    p = run([str(TU / "m1_feats.py"), "--dataset", "hwu64", "--table", "potion", "--arms", "potion:none", "--out-root", str(out),
             "--device", "cpu", "--load", f"{STANDIN}:load_small"], timeout=1800)
    check(f"feats potion: exit 0 ({time.time() - t_0:.0f} s)", p.returncode == 0, p.stderr[-800:])
    if p.returncode == 0:
        zp = np.load(out / "potion" / "hwu64__p32tok_K512.npz")
        mp = json.loads((out / "potion" / "hwu64__p32tok_K512.meta.json").read_text())
        check("feats potion: the table is named, test rows = val rows, 512 concepts",
              mp["table"] == "potion-base-32M" and np.array_equal(zp["test__exists"], zp["val__exists"])
              and zp["train__exists"].shape[1] == 512, mp["table"])
    m1o = RUNS / "m1"
    m1o.mkdir()
    wrap = RUNS / "nosync.py"
    wrap.write_text("import runpy\nimport sys\n\nimport torch\n\ntorch.cuda.synchronize = lambda *a, **k: None  # no GPU in this test\n"
                    "sys.argv = sys.argv[1:]\nrunpy.run_path(sys.argv[0], run_name='__main__')\n")
    tm_args = [str(wrap), str(TU / "m1_tm.py"), "--dataset", "hwu64", "--view", "existsT", "--seed", "7", "--s", "8",
               "--emb-dir", str(out / "a3B"), "--tag", "t", "--out-dir", str(m1o), "--device", "cpu", "--smoke",
               "--load", f"{STANDIN}:load_small"]
    t_0 = time.time()
    p = run(tm_args, timeout=1800)
    check(f"m1_tm smoke: exit 0 ({time.time() - t_0:.0f} s)", p.returncode == 0, p.stderr[-800:])
    if p.returncode == 0:
        rr = [json.loads(x) for x in (m1o / "results_m1_tm.jsonl").read_text().splitlines()]
        r0 = rr[0] if rr else {}
        pr = np.load(r0["preds"]) if rr else np.zeros(0)
        check("m1_tm smoke: one row, marked smoke, 'test' accuracy = val accuracy (the stand-in), predictions saved",
              len(rr) == 1 and r0["smoke"] and r0["test_acc"] == r0["val_acc_check"] == r0["val_acc"]
              and pr.shape[0] == len(ds.texts["val"]) and r0["n_train"] == 1500, str({k: r0.get(k) for k in ("test_acc", "val_acc", "val_acc_check")}))
        check("m1_tm smoke: the row has utc and the sha256 of the code and the cache",
              "utc" in r0 and set(r0["sha256"]) == {"m1_tm.py", "exists_tm.py", "cache"})
        p2 = run(tm_args, timeout=1800)
        check("m1_tm: a second run with the same key is refused (one test read per run)",
              p2.returncode != 0 and "one test read per run" in p2.stderr, p2.stderr[-300:])
    lr_args = [str(TU / "m1_lr.py"), "--name", "t", "--dataset", "hwu64", "--emb-dir", str(out / "a3B"), "--out-dir", str(m1o),
               "--device", "cpu", "--load", f"{STANDIN}:load_small"]
    t_0 = time.time()
    p = run(lr_args, timeout=1800)
    check(f"m1_lr: exit 0 ({time.time() - t_0:.0f} s)", p.returncode == 0, p.stderr[-800:])
    if p.returncode == 0:
        lrr = [json.loads(x) for x in (m1o / "results_m1_lr.jsonl").read_text().splitlines()]
        l0 = lrr[0] if lrr else {}
        check("m1_lr: one row, 'test' accuracy = val accuracy at the picked C, every C's val accuracy kept",
              len(lrr) == 1 and l0["lr_test"] == l0["lr_val"] and set(l0["acc_by_C"]) == {"1.0", "10.0", "100.0"}
              and l0["lr_val"] == max(l0["acc_by_C"].values()) and "m1_lr.py" in l0["sha256"], str(l0.get("acc_by_C")))
        p2 = run(lr_args, timeout=1800)
        check("m1_lr: a second run is refused (one test read per arm)", p2.returncode != 0 and "one test read per arm" in p2.stderr,
              p2.stderr[-300:])
else:
    print("feats part not run (no --feats)")

shutil.rmtree(RUNS / "verdict2", ignore_errors=True)
print(f"m1 tests: {results['pass']} passed, {results['fail']} failed")
sys.exit(0 if results["fail"] == 0 else 1)
