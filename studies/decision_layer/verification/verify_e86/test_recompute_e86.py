"""Test of recompute_e86.py on a made-up study folder (no real data).

    OMP_NUM_THREADS=1 <python with numpy> -I test_recompute_e86.py
"""
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
N, K = 400, 6
TIMES = {"e84": "2026-10-22T10:37:12Z", "e86": "2026-10-23T00:58:18Z", "note": "2026-10-25T10:52:22Z"}
SPEC = __import__("importlib.util").util.spec_from_file_location("rc", os.path.join(HERE, "recompute_e86.py"))
RC = __import__("importlib.util").util.module_from_spec(SPEC)
SPEC.loader.exec_module(RC)


def labels(ds):
    return np.random.default_rng({"multiwoz": 1, "star": 2, "abcd": 3}[ds]).integers(0, K, N)


def preds_with(y, c):
    p = (y + 1) % K
    p[:c] = y[:c]
    return p


def build(root, acc, extra_rows=()):
    """acc[(ds, cell)] = (best correct, last-10 correct) of N, the same for both seeds."""
    for d in ("logs", "results/explore/e84_preds", "results/explore/e86_preds"):
        os.makedirs(os.path.join(root, d), exist_ok=True)
    for k, p in RC.FREEZES.items():
        json.dump({"frozen_utc": TIMES[k]}, open(os.path.join(root, p), "w"))
    rows = []
    for ds, cs in RC.CELLS.items():
        y = labels(ds)
        os.makedirs(os.path.join(root, "data", ds), exist_ok=True)
        with gzip.open(os.path.join(root, "data", ds, "examples_dev.json.gz"), "wt") as f:
            json.dump({"rows": [{"y": int(v), "convo_id": f"c{i // 8}"} for i, v in enumerate(y)]}, f)
        lrd = os.path.join(root, "results", "explore", "dryrun", RC.LR[ds])
        os.makedirs(lrd, exist_ok=True)
        json.dump({"arm": "lr", "dev_acc": 0.5}, open(os.path.join(lrd, "seed0.json"), "w"))
        for cell, (tag, folder, rec) in cs.items():
            best, l10 = acc[(ds, cell)]
            for sd in RC.SEEDS:
                ev = [l10 - 3] * 10 + [l10] * 10
                P = np.stack([preds_with(y, c) for c in ev] + [preds_with(y, best)]).astype(np.int16)
                np.savez_compressed(os.path.join(root, "results", "explore", folder, f"{tag}_seed{sd}.npz"), preds=P, y_val=y.astype(np.int16))
                t = {"e84": "2026-10-22T12:00:00Z", "e86": "2026-10-23T02:00:00Z", "note": "2026-10-25T12:00:00Z"}[rec]
                rows.append({"tag": tag, "seed": sd, "arm": "tm", "time": t, "acc": best / N,
                             "history": [{"epoch": 2 * (i + 1), "val_acc": c / N} for i, c in enumerate(ev)], "cfg": {}})
    rows += list(extra_rows)
    with open(os.path.join(root, "results", "explore", "abcd.jsonl"), "w") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows))


def run(root):
    out = os.path.join(root, "out.json")
    p = subprocess.run([sys.executable, "-I", os.path.join(HERE, "recompute_e86.py"), root, "--out", out], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    return json.load(open(out))


def base(drop, one, ref=(200, 190)):
    acc = {}
    for ds, cs in RC.CELLS.items():
        acc[(ds, "ref")] = ref
        acc[(ds, "drop")] = drop[ds]
        if "one" in cs:
            acc[(ds, "one")] = one[ds]
    return acc


def main():
    tmp = tempfile.mkdtemp(prefix="rc86_")
    try:
        # kept everywhere at exactly +0.5 (2 of 400 rows), one setting at exactly -0.5 against STAR's kept drop cell
        r1 = os.path.join(tmp, "a")
        build(r1, base({"multiwoz": (202, 192), "star": (204, 194), "abcd": (202, 192)}, {"star": (202, 192), "abcd": (202, 192)}))
        o = run(r1)
        assert o["kept_on"] == ["multiwoz", "star", "abcd"] and o["E86.1"] == "MET", o
        assert abs(o["star_one_minus_best"]["best_dev_points"] + 0.5) < 1e-9 and o["star_one_minus_best"]["holds"], o["star_one_minus_best"]
        assert o["multiwoz_one_minus_best"]["best_dev_points"] == 0 and o["E86.3"] == "MET", o
        assert abs(o["cells"]["abcd/drop"]["tm_minus_lr_points"] - 0.5) < 1e-9
        lo, hi = o["abcd_drop_minus_ref"]["boot_best_dev_points"]
        assert lo <= 0.5 <= hi, (lo, hi)
        assert not o["problems"]
        # kept on MultiWOZ only; STAR's one setting far below
        r2 = os.path.join(tmp, "b")
        build(r2, base({"multiwoz": (204, 194), "star": (201, 191), "abcd": (200, 196)}, {"star": (150, 140), "abcd": (200, 190)}))
        o = run(r2)
        assert o["kept_on"] == ["multiwoz"] and o["E86.1"] == "not met" and o["E86.3"] == "not met", o
        assert o["abcd_drop_minus_ref"]["kept"] is False  # best-dev +0.0, last-10 +1.5: not kept
        # a stale STAR row from before the note is not eligible; two eligible rows are a problem
        r3 = os.path.join(tmp, "c")
        stale = {"tag": "e86_star_C4_c16000_s16_ref", "seed": 1, "arm": "tm", "time": "2026-10-23T02:26:20Z", "acc": 0.1, "history": [], "cfg": {}}
        dup = {"tag": "e86_abcd_C4_c4000_s16_ref", "seed": 2, "arm": "tm", "time": "2026-10-23T03:00:00Z", "acc": 0.5, "history": [], "cfg": {}}
        build(r3, base({"multiwoz": (202, 192), "star": (204, 194), "abcd": (202, 192)}, {"star": (202, 192), "abcd": (202, 192)}), [stale])
        o = run(r3)
        assert not o["problems"] and o["E86.1"] == "MET", o["problems"]
        build(os.path.join(tmp, "d"), base({"multiwoz": (202, 192), "star": (204, 194), "abcd": (202, 192)}, {"star": (202, 192), "abcd": (202, 192)}), [dup])
        o = run(os.path.join(tmp, "d"))
        assert any("2 eligible rows" in p for p in o["problems"]), o["problems"]
        print("test_recompute_e86: 4 cases passed (kept at exactly +0.5 and E86.3 at exactly -0.5; MultiWOZ only; stale STAR row ignored;"
              " duplicate row flagged)")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
