"""PLAN_logsize_dev.md: agent_preds.py, unchanged, trained on the rows of a pool of N training dialogues (dev only).

usage (decision_layer's .venv-dl):
  python -I logsize_agent.py --size 64 [--subsets runs/logsize/subsets.json] -- \
      --dataset abcd --encoder none --table c4words-S1000-v1 --parts mem,words,xcust,xrecent --arms tm \
      --clauses 4000 --s 16 --seed 1 --out runs/logsize/agent_abcd_N64
Everything after "--" goes to agent_preds.main unchanged. The one change: dl.abcd.load_examples("train", ...) returns
only the rows whose convo_id is in the pool (--size whole: every row, so the run must equal agent_preds.py's). Dev
rows are untouched. The test split is refused. Writes OUT/pool.json beside agent_preds' own files.
"""
import argparse
import hashlib
import json
import os
import sys
import time

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
import logsize_subsets as S  # noqa: E402

REFUSED = ("--test-registration", "--dry-run-dev")


def split_argv(argv):
    if "--" not in argv:
        raise SystemExit("usage: logsize_agent.py --size N [--subsets FILE] -- <agent_preds.py arguments>")
    k = argv.index("--")
    return argv[:k], argv[k + 1:]


def install(keep):
    """Patch dl.abcd.load_examples so that the train split keeps only the pool's rows; returns the module."""
    import agent_preds  # noqa: F401 - puts decision_layer/src on sys.path
    import dl.abcd as AB
    orig = AB.load_examples

    def load_examples(split, dataset="abcd"):
        rows, buttons = orig(split, dataset)
        if split == "train" and keep is not None:
            rows = [r for r in rows if r["convo_id"] in keep]
        return rows, buttons

    load_examples.orig = orig
    AB.load_examples = load_examples
    return AB


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    own, rest = split_argv(argv)
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True, help="64, 256, 1024, 4096 or whole")
    ap.add_argument("--subsets", default=os.path.join(H, "runs/logsize/subsets.json"))
    a = ap.parse_args(own)
    if any(x.split("=")[0] in REFUSED for x in rest):
        raise SystemExit("refused: logsize_agent.py is dev only")
    pa = argparse.ArgumentParser(add_help=False)
    pa.add_argument("--dataset", required=True)
    pa.add_argument("--out", required=True)
    b, _ = pa.parse_known_args(rest)
    with open(a.subsets, "rb") as h:
        raw = h.read()
    subsets = json.loads(raw)
    keep = S.keep_ids(subsets, b.dataset, a.size)
    AB = install(keep)
    tr, _ = AB.load_examples("train", b.dataset)
    os.makedirs(b.out, exist_ok=True)
    info = {"plan": "PLAN_logsize_dev.md", "dataset": b.dataset, "size": a.size,
            "n_dialogues": len({r["convo_id"] for r in tr}), "n_train_rows": len(tr),
            "subsets": a.subsets, "subsets_sha256": hashlib.sha256(raw).hexdigest(),
            "argv": [sys.argv[0]] + argv, "utc_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(os.path.join(b.out, "pool.json"), "w") as h:
        json.dump(info, h, indent=1)
    print("pool", json.dumps({k: info[k] for k in ("dataset", "size", "n_dialogues", "n_train_rows")}), flush=True)
    import agent_preds as A
    return A.main(rest)


if __name__ == "__main__":
    sys.exit(main())
