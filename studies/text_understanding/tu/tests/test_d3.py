#!/usr/bin/env python
"""Tests for 3d's code (tu/PLAN_3d.md). A plain script, no pytest; CPU only, no GPU. No test split is read: the runs on
real data use m1_standin.load_small (train cut to 1,500 rows), and the 3d scripts never take a test split.

    OMP_NUM_THREADS=2 nice -n 19 $PY -I -B tu/tests/test_d3.py [--feats]

- The self-tests of d3_pairs.py, d3_fit.py (planted word and pair rows recovered; the no-pair tokens are 3a's rows; a
  negative check) and d3_mw.py.
- d3_fit.pair_list on the real counts file, against an independent count of the same rule.
- d3_feats.PairTable on a tiny table: the tokens with and without pairs, the coverage counts.
- d3_verdict.py on synthetic rows: a bar met exactly at its boundary, a miss by one probe, the descriptive lines, the
  M1 reproduction line, and the refusals (a missing seed, a smoke row, s 16, a doubled LR row, an overwrite).
- With --feats (a few minutes): d3_feats.py on the CPU for HWU64 (stand-in), with a3B's table and a synthetic pair
  table (a3B's word rows; a pair row = the unit mean of its two words' rows, test only):
  - the a3B arm equals tu/m1_feats.py's a3B arm on the same stand-in, array for array, with the same k-means;
  - JnoP (pair tokens off) equals the a3B arm; Jwc (word-only k-means) shares JnoP's k-means and its exists values are
    never lower (a max over more tokens), and higher somewhere; J clusters pair types too;
  - d3_exists_tm.py --smoke and d3_lr.py: one row each, val only; a cache with test rows is refused;
  - network-free at run time: after each feature build and the TM smoke, no transformers, sentence_transformers or
    model2vec module is loaded.
  The TM's fit calls torch.cuda.synchronize() unconditionally, so d3_exists_tm.py runs through a wrapper that makes
  that call a no-op (this test only).
Each run has its own folder in tu/tests/runs/<UTC stamp>_d3/.
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))

import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

TU = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TU))
sys.path.insert(0, str(TU.parent))
sys.path.insert(0, str(TU.parent / "w1"))
sys.path.insert(0, f"{_REPO_ROOT}/studies/decision_layer/src")
PY = sys.executable
RUNS = TU / "tests" / "runs" / (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_d3")
STANDIN = str(TU / "m1_standin.py")
results = {"pass": 0, "fail": 0}


def check(name: str, ok: bool, detail: str = "") -> None:
    results["pass" if ok else "fail"] += 1
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f": {detail}"))


def run(args: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run([PY, "-I", "-B", *args], capture_output=True, text=True, timeout=timeout, check=False)


RUNS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- self-tests
for name in ("d3_pairs.py", "d3_fit.py", "d3_mw.py"):
    p = run([str(TU / name), "--selftest"], timeout=900)
    check(f"{name} --selftest", p.returncode == 0 and "SELFTEST PASSED" in p.stdout, (p.stdout + p.stderr)[-600:])

# ---------------------------------------------------------------- the pair list against an independent count
import d3_fit as DF

z = np.load(DF.TABLE)
words = [str(w) for w in z["words"]]
w1, w2, c = DF.pair_list(words)
q = np.load(DF.COUNTS)
alnum = re.compile(r"^(?:[a-z0-9]+|n't|'[a-z]+)$")
indep = [(int(a), int(b)) for a, b, n in zip(q["w1"], q["w2"], q["counts"])
         if n >= 20 and alnum.match(words[int(a)]) and alnum.match(words[int(b)])]
check(f"pair_list: {len(w1):,} pairs, the same list and order as an independent count of the rule",
      [(int(a), int(b)) for a, b in zip(w1, w2)] == indep, f"{len(w1)} vs {len(indep)}")
check("pair_list: every count >= 20, sorted descending, no punctuation token",
      bool((c >= 20).all()) and bool((np.diff(c) <= 0).all())
      and not any(not alnum.match(words[int(a)]) or not alnum.match(words[int(b)]) for a, b in zip(w1[:5000], w2[:5000])))
check("pair_list: the counts file was made from the same word list", int(q["V"]) == len(words))

# ---------------------------------------------------------------- PairTable on a tiny table
import d3_feats as DFe

tw = ["turn", "off", "the", "lights", "on", ","]
rng = np.random.default_rng(3)
tiny = RUNS / "tiny_table.npz"
np.savez(tiny, codes=rng.normal(size=(6, 4)).astype(np.float16), words=np.array(tw), counts=np.ones(6), vocab_ids=np.arange(6),
         pair_codes=rng.normal(size=(2, 4)).astype(np.float16), pair_w1=np.array([0, 2], np.int32),
         pair_w2=np.array([1, 3], np.int32), pair_counts=np.array([50, 40]))
pt_on, pt_off = DFe.PairTable(tiny, True), DFe.PairTable(tiny, False)
s = "Turn off the lights, on"
check("PairTable: words, then the listed pairs as rows V + j", pt_on.ids([s]) == [[0, 1, 2, 3, 5, 4, 6, 7]], str(pt_on.ids([s])))
check("PairTable: pair tokens off = words only", pt_off.ids([s]) == [[0, 1, 2, 3, 5, 4]] and pt_off.M.shape == (6, 4))
check("PairTable: token names", pt_on.token(6) == "turn_off" and pt_on.token(7) == "the_lights" and pt_on.token(0) == "turn")
cv = pt_on.coverage([s, "on the lights", "zzz"])
check("PairTable: coverage counts", cv == {"texts": 3, "adjacent_word_pairs": 7, "listed": 3, "listed_share": round(3 / 7, 4),
                                           "texts_with_a_pair_token": 2}, str(cv))
check("PairTable: order matters ('off turn' lists nothing)", pt_on.ids(["off turn"]) == [[1, 0]])

# ---------------------------------------------------------------- d3_verdict.py on synthetic rows
SETS, ARMS, SEEDS = ("clinc150", "banking77", "hwu64"), ("a3B", "J", "Jwc", "JnoP", "MW"), (7, 8, 9)
CFG = {"n_clauses": 4000, "T": 4000, "s": 8.0, "epochs": 40, "batch_size": 128}
BASE = {"clinc150": 0.9000, "banking77": 0.8800, "hwu64": 0.8600}
GAIN = {"clinc150": 0.0050, "banking77": 0.0050, "hwu64": 0.0010}  # J - a3B: exactly +0.5 on two sets


def synth(d0: Path, lr_gain: dict | None = None, tweak=None) -> None:
    d0.mkdir(parents=True, exist_ok=True)
    tm, lr = [], []
    for d in SETS:
        for i, a in enumerate(ARMS):
            off = {"a3B": 0.0, "J": GAIN[d], "Jwc": 0.002, "JnoP": 0.001, "MW": 0.003}[a]
            for s_ in SEEDS:
                tm.append({"tag": "d3", "dataset": d, "arm": a, "seed": s_, "smoke": False, "view": "existsT", "repeat": 1,
                           "cfg": dict(CFG), "val_acc": round(BASE[d] + off + 0.0001 * (s_ - 8), 4)})
            g = (lr_gain or GAIN)[d] if a == "J" else off
            lr.append({"dataset": d, "name": a, "arm": a, "lr_val": round(BASE[d] + 0.01 + g, 4), "C": 10.0, "i": i})
    if tweak:
        tweak(tm, lr)
    (d0 / "results_d3_val.jsonl").write_text("".join(json.dumps(r) + "\n" for r in tm))
    (d0 / "results_d3_lr.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lr))


m1d = RUNS / "m1"
m1d.mkdir()
m1rows = [{"tag": "m1", "dataset": d, "arm": "a3B", "repeat": 1, "seed": s_, "val_acc": round(BASE[d] + 0.0001 * (s_ - 8), 4)}
          for d in SETS for s_ in SEEDS]
m1rows += [{"tag": "m1", "dataset": "clinc150", "arm": "a3B", "repeat": 5, "seed": 7, "val_acc": 0.1}]
(m1d / "results_m1_tm.jsonl").write_text("".join(json.dumps(r) + "\n" for r in m1rows))


def verdict(d0: Path) -> subprocess.CompletedProcess:
    return run([str(TU / "d3_verdict.py"), "--d3", str(d0), "--m1", str(m1d)])


v1 = RUNS / "verdict1"
synth(v1)
p = verdict(v1)
vj = json.loads((v1 / "verdict_d3.json").read_text()) if p.returncode == 0 else {}
check("verdict: PASS with the bar met exactly at +0.5 on two sets (both probes)",
      p.returncode == 0 and vj.get("verdict") == "PASS" and vj.get("sets_met") == 2
      and vj["sets"]["clinc150"]["met"] and vj["sets"]["banking77"]["met"] and not vj["sets"]["hwu64"]["met"], p.stdout[-500:])
check("verdict: the gains, the per-seed values and the descriptive lines",
      p.returncode == 0 and vj["sets"]["hwu64"]["J_minus_a3B"] == {"tm": 0.1, "lr": 0.1}
      and vj["sets"]["clinc150"]["tm_per_seed"] == {"7": 0.5, "8": 0.5, "9": 0.5}
      and vj["sets"]["clinc150"]["descriptive"]["MW_minus_a3B"] == {"tm": 0.3, "lr": 0.3}
      and vj["sets"]["clinc150"]["descriptive"]["J_minus_MW"] == {"tm": 0.2, "lr": 0.2}, json.dumps(vj.get("sets", {}).get("hwu64", {}))[:400])
check("verdict: the a3B reference equals M1's a3B val rows (repeat 5 rows ignored)",
      p.returncode == 0 and all(vj["sets"][d]["a3B_equals_M1_val"] for d in SETS))
check("verdict: the text says PASS and 2 of 3", "VERDICT 3d (val): PASS" in p.stdout and "on 2 of 3 sets" in p.stdout, p.stdout[:300])
p2 = verdict(v1)
check("verdict: refuses to overwrite", p2.returncode != 0 and "exists" in p2.stderr, p2.stderr[-200:])
v2 = RUNS / "verdict2"
synth(v2, lr_gain={"clinc150": 0.0050, "banking77": 0.0049, "hwu64": 0.0010})
p = verdict(v2)
vj2 = json.loads((v2 / "verdict_d3.json").read_text()) if p.returncode == 0 else {}
check("verdict: FAIL when one probe misses by 0.01 on the second set", p.returncode == 0 and vj2.get("verdict") == "FAIL"
      and vj2.get("sets_met") == 1 and not vj2["sets"]["banking77"]["met"], p.stdout[-300:])


def refused(name: str, tweak, needle: str) -> None:
    d0 = RUNS / f"ref_{name}"
    synth(d0, tweak=tweak)
    p = verdict(d0)
    check(f"verdict refuses: {name}", p.returncode == 2 and needle in p.stdout and not (d0 / "verdict_d3.json").exists(),
          p.stdout[-300:])


refused("a missing seed", lambda tm, lr: tm.pop(0), "missing TM row")
refused("a smoke row", lambda tm, lr: tm[3].update(smoke=True), "smoke")
refused("s 16", lambda tm, lr: tm[5]["cfg"].update(s=16.0), "cfg")
refused("a doubled LR row", lambda tm, lr: lr.append(dict(lr[0])), "doubled LR row")
refused("an extra seed", lambda tm, lr: tm.append(dict(tm[0], seed=4)), "other seeds")

# ---------------------------------------------------------------- --feats: the real code paths on the CPU
if "--feats" in sys.argv:
    import m1_standin as S

    ds = S.load_small("hwu64")
    A3B = TU / "a3" / "tables" / "a3B_d768.npz"
    za = np.load(A3B)
    U = za["codes"].astype(np.float32)
    k = 3000
    m = U[w1[:k]] + U[w2[:k]]
    syn = RUNS / "syn_pairs.npz"
    np.savez(syn, codes=za["codes"], words=za["words"], counts=za["counts"], vocab_ids=za["vocab_ids"],
             pair_codes=(m / np.linalg.norm(m, axis=1, keepdims=True)).astype(np.float16), pair_w1=w1[:k].astype(np.int32),
             pair_w2=w2[:k].astype(np.int32), pair_counts=c[:k])
    out = RUNS / "feats"
    common = ["--dataset", "hwu64", "--out-root", str(out), "--device", "cpu", "--load", f"{STANDIN}:load_small"]
    t_0 = time.time()
    p = run([str(TU / "m1_feats.py"), "--table", f"wordtab:{A3B}", "--arms", "m1a3B:none", *common], timeout=1800)
    check(f"m1_feats a3B (the reference): exit 0 ({time.time() - t_0:.0f} s)", p.returncode == 0, p.stderr[-600:])
    NET = "\nbad = sorted(m for m in sys.modules if m.split('.')[0] in ('transformers', 'sentence_transformers', 'model2vec'))\nprint('NETWORK_MODULES', bad)\n"
    nonet = RUNS / "nonet.py"
    nonet.write_text("import runpy\nimport sys\n\nsys.argv = sys.argv[1:]\nrunpy.run_path(sys.argv[0], run_name='__main__')" + NET)
    tf = RUNS / "imports_tf.py"
    tf.write_text("import transformers  # noqa: F401\n")
    p = run([str(nonet), str(tf)], timeout=600)
    check("network check: it sees a loaded transformers module (negative check)",
          p.returncode == 0 and "NETWORK_MODULES [" in p.stdout and "'transformers'" in p.stdout, p.stdout[-300:] + p.stderr[-300:])
    arms = {"a3B": (A3B, "off", "all"), "JnoP": (syn, "off", "all"), "Jwc": (syn, "on", "words"), "J": (syn, "on", "all")}
    for arm, (tab, pt, km) in arms.items():
        t_0 = time.time()
        p = run([str(nonet), str(TU / "d3_feats.py"), "--arm", arm, "--table", str(tab), "--pair-tokens", pt, "--kmeans", km,
                 *common], timeout=1800)
        check(f"d3_feats {arm}: exit 0 ({time.time() - t_0:.0f} s); no network module loaded",
              p.returncode == 0 and "NETWORK_MODULES []" in p.stdout, (p.stdout[-300:] + p.stderr[-600:]))
    f = {a: np.load(out / a / "hwu64__p32tok_K512.npz") for a in ("m1a3B", *arms)}
    mt = {a: json.loads((out / a / "hwu64__p32tok_K512.meta.json").read_text()) for a in ("m1a3B", *arms)}
    keys = [x for x in ("train__exists", "val__exists", "train__mean", "val__mean", "oos_val__exists", "y_train", "y_val")
            if x in f["m1a3B"].files]
    check("feats: the a3B arm equals m1_feats.py's, array for array, with the same k-means",
          all(np.array_equal(f["a3B"][x], f["m1a3B"][x]) for x in keys) and mt["a3B"]["kmeans_run_id"] == mt["m1a3B"]["kmeans_run_id"]
          and mt["a3B"]["concepts_sha16"] == mt["m1a3B"]["concepts_sha16"], mt["a3B"]["kmeans_run_id"])
    check("feats: no test split in the 3d caches", not any(x.startswith("test__") or x == "y_test" for x in f["J"].files)
          and "test" not in mt["J"]["splits"], str(mt["J"]["splits"]))
    check("feats: JnoP (pair tokens off) equals the a3B arm",
          all(np.array_equal(f["JnoP"][x], f["a3B"][x]) for x in keys) and mt["JnoP"]["train_vocab_pairs"] == 0)
    ge = bool((f["Jwc"]["train__exists"] >= f["JnoP"]["train__exists"]).all())
    gt = bool((f["Jwc"]["train__exists"] > f["JnoP"]["train__exists"]).any())
    check("feats: Jwc shares JnoP's word-only k-means; its exists values are never lower and higher somewhere",
          mt["Jwc"]["kmeans_run_id"] == mt["JnoP"]["kmeans_run_id"] and ge and gt and mt["Jwc"]["train_vocab_pairs"] == 0,
          f"{mt['Jwc']['kmeans_run_id']} {mt['JnoP']['kmeans_run_id']} ge {ge} gt {gt}")
    cj = mt["J"]["coverage"]["train"]
    check(f"feats: J clusters word and pair types ({mt['J']['train_vocab_words']} words, {mt['J']['train_vocab_pairs']} pairs), "
          f"coverage listed {cj['listed']:,} of {cj['adjacent_word_pairs']:,}",
          mt["J"]["train_vocab_pairs"] > 0 and mt["J"]["train_vocab_words"] == mt["JnoP"]["train_vocab_words"]
          and 0 < cj["listed"] <= cj["adjacent_word_pairs"] and mt["J"]["pair_tokens"] and mt["J"]["kmeans_types"] == "all")
    d3o = RUNS / "d3"
    d3o.mkdir()
    wrap = RUNS / "nosync.py"
    wrap.write_text("import runpy\nimport sys\n\nimport torch\n\ntorch.cuda.synchronize = lambda *a, **k: None  # no GPU in this test\n"
                    "sys.argv = sys.argv[1:]\nrunpy.run_path(sys.argv[0], run_name='__main__')" + NET)
    tm_args = [str(wrap), str(TU / "d3_exists_tm.py"), "--dataset", "hwu64", "--view", "existsT", "--seed", "7", "--s", "8",
               "--emb-dir", str(out / "J"), "--tag", "t", "--out-dir", str(d3o), "--device", "cpu", "--smoke",
               "--load", f"{STANDIN}:load_small"]
    t_0 = time.time()
    p = run(tm_args, timeout=1800)
    check(f"d3_exists_tm smoke: exit 0 ({time.time() - t_0:.0f} s); no network module loaded",
          p.returncode == 0 and "NETWORK_MODULES []" in p.stdout, p.stdout[-300:] + p.stderr[-800:])
    if p.returncode == 0:
        rr = [json.loads(x) for x in (d3o / "results_d3_val.jsonl").read_text().splitlines()]
        r0 = rr[0] if rr else {}
        check("d3_exists_tm smoke: one row, study 3d, marked smoke, val only (no test field), 1,500 training rows",
              len(rr) == 1 and r0["study"] == "3d" and r0["smoke"] and "test_acc" not in r0 and "preds" not in r0
              and r0["n_train"] == 1500 and r0["val_acc_check"] == r0["val_acc"] and r0["arm"] == "J",
              str({x: r0.get(x) for x in ("study", "val_acc", "val_acc_check", "n_train")}))
        check("d3_exists_tm smoke: the row has utc, the key and the sha256 of the code and the cache",
              "utc" in r0 and r0["key"] == "t_J_hwu64_r1_s7" and set(r0["sha256"]) == {"d3_exists_tm.py", "exists_tm.py", "cache"})
    lr_args = [str(TU / "d3_lr.py"), "--name", "t", "--dataset", "hwu64", "--emb-dir", str(out / "J"), "--out-dir", str(d3o),
               "--device", "cpu", "--load", f"{STANDIN}:load_small"]
    t_0 = time.time()
    p = run(lr_args, timeout=1800)
    check(f"d3_lr: exit 0 ({time.time() - t_0:.0f} s)", p.returncode == 0, p.stderr[-800:])
    if p.returncode == 0:
        lrr = [json.loads(x) for x in (d3o / "results_d3_lr.jsonl").read_text().splitlines()]
        l0 = lrr[0] if lrr else {}
        check("d3_lr: one row, val only, every C's val accuracy kept, the picked C the best",
              len(lrr) == 1 and "lr_test" not in l0 and set(l0["acc_by_C"]) == {"1.0", "10.0", "100.0"}
              and l0["lr_val"] == max(l0["acc_by_C"].values()) and "d3_lr.py" in l0["sha256"] and l0["study"] == "3d",
              str(l0.get("acc_by_C")))
    bad = RUNS / "badcache"
    bad.mkdir()
    zj = dict(np.load(out / "J" / "hwu64__p32tok_K512.npz"))
    zj["test__exists"] = zj["val__exists"]
    np.savez(bad / "hwu64__p32tok_K512.npz", **zj)
    (bad / "hwu64__p32tok_K512.meta.json").write_text((out / "J" / "hwu64__p32tok_K512.meta.json").read_text())
    p = run([str(wrap), str(TU / "d3_exists_tm.py"), "--dataset", "hwu64", "--view", "existsT", "--seed", "7", "--emb-dir", str(bad),
             "--tag", "t", "--out-dir", str(d3o), "--device", "cpu", "--smoke", "--load", f"{STANDIN}:load_small"], timeout=600)
    check("d3_exists_tm: a cache with test rows is refused", p.returncode != 0 and "no test rows" in p.stderr, p.stderr[-300:])
    p = run([str(TU / "d3_lr.py"), "--name", "t2", "--dataset", "hwu64", "--emb-dir", str(bad), "--out-dir", str(d3o),
             "--device", "cpu", "--load", f"{STANDIN}:load_small"], timeout=600)
    check("d3_lr: a cache with test rows is refused", p.returncode != 0 and "no test rows" in p.stderr, p.stderr[-300:])
else:
    print("feats part not run (no --feats)")

print(f"d3 tests: {results['pass']} passed, {results['fail']} failed")
sys.exit(0 if results["fail"] == 0 else 1)
