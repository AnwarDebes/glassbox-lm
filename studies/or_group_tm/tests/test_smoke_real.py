"""End-to-end smoke tests of src/run_slide10.py and src/run_wiki.py on tiny settings (a few hundred examples,
few clauses, 2 passes). They check that the runners write their files and refuse the held-back splits.

Run from the study folder:
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 .venv/bin/python -I tests/test_smoke_real.py
Writes its run folders under logs/tests/smoke_runs/ and leaves the run logs alone.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import run_slide10
import run_wiki

OUT = os.path.join(ROOT, "logs", "tests", "smoke_runs")


def sizes():
    logs = [os.path.join(ROOT, "logs", f) for f in ("RUNS.md", "RUNS_slide10.md", "RUNS_wiki.md")]
    return [(os.path.getsize(f) if os.path.exists(f) else None) for f in logs]


def refused(fn, argv):
    try:
        fn(argv)
    except SystemExit as e:
        return e.code != 0
    return False


def main():
    before = sizes()
    for version in ("text", "pairs"):
        run_dir, final = run_slide10.main(["--version", version, "--seed", "990", "--clauses", "6", "--K", "8", "--passes", "2",
                                           "--n-train", "400", "--name", f"smoke_slide10_{version}", "--out-root", OUT, "--no-log"])
        for f in ("config.json", "metrics.json", "include_bits.npz", "check2b_dev.txt", "state.npz"):
            assert os.path.getsize(os.path.join(run_dir, f)) > 0, f
        assert 0.0 <= final["dev_plain_acc"] <= 1.0 and "coverage_dev_288" in final["check2b"]
        z = np.load(os.path.join(run_dir, "include_bits.npz"))
        assert int(z["K"]) == 8 and int(z["F"]) == 221
        print("ok   slide10", version, os.path.basename(run_dir), flush=True)
    assert refused(run_slide10.main, ["--version", "text", "--seed", "1", "--split", "eval", "--name", "x", "--out-root", OUT, "--no-log"]), \
        "the eval sentences must be refused without a frozen record"
    run_dir, final = run_wiki.main(["--data", "sa", "--seed", "990", "--clauses", "30", "--passes", "2", "--n-train", "300",
                                    "--n-dev", "200", "--name", "smoke_wiki_sa", "--out-root", OUT, "--no-log"])
    for f in ("config.json", "metrics.json", "groups.txt", "state.npz", "scores_dev.npz"):
        assert os.path.getsize(os.path.join(run_dir, f)) > 0, f
    with open(os.path.join(run_dir, "metrics.json")) as f:
        m = json.load(f)
    assert len(m["history"]) == 2 and m["final"]["n_train"] == 300 and m["final"]["n_eval"] == 200
    assert 0.0 <= m["final"]["dev_auc"] <= 1.0 and 0.0 <= m["final"]["partner_shuffle_auc"] <= 1.0
    assert np.load(os.path.join(run_dir, "state.npz"))["state"].shape == (30, 2, 1024)
    assert len(np.load(os.path.join(run_dir, "scores_dev.npz"))["votes"]) == 200
    print("ok   wiki sa", os.path.basename(run_dir), flush=True)
    assert refused(run_wiki.main, ["--data", "sa", "--seed", "1", "--split", "test", "--name", "x", "--out-root", OUT, "--no-log"]), \
        "the test pairs must be refused without a frozen record"
    assert refused(run_wiki.main, ["--data", "ns_all", "--F", "3072", "--seed", "1", "--name", "x", "--out-root", OUT, "--no-log"])
    assert sizes() == before, "--no-log must leave the run logs alone"
    print("SMOKE TEST PASSED (slide10 text, slide10 pairs, wiki sa, 3 refusals)")


if __name__ == "__main__":
    main()
