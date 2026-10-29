"""PLAN_logsize_dev.md: llm_next_action.py, unchanged, retrieving its solved examples from a pool of N training
dialogues (dev only).

usage (decision_layer's .venv-dl):
  python -I logsize_llm.py --size 64 [--subsets runs/logsize/subsets.json] -- \
      --dataset abcd --n 500 --k 50 --seed 2 --exclude runs/llm_q1_dev/abcd/preds.jsonl \
      --url http://127.0.0.1:18083 --out runs/logsize/llm_N64/abcd
Everything after "--" goes to llm_next_action.main unchanged. The one change: rows_of(dataset, "train") returns only
the rows whose convo_id is in the pool (--size whole: every row). The prompt, the action list, the written policy and
the drawn dev rows are untouched. The test split is refused. Writes OUT/pool.json beside the client's own files.
"""
import argparse
import hashlib
import json
import os
import sys
import time

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
import llm_next_action as L  # noqa: E402
import logsize_subsets as S  # noqa: E402

REFUSED = ("--test-registration",)


def install(keep):
    orig = getattr(L.rows_of, "orig", L.rows_of)

    def rows_of(dataset, split, allow_test=False):
        rows, buttons = orig(dataset, split, allow_test)
        if split == "train" and keep is not None:
            rows = [r for r in rows if r["convo_id"] in keep]
        return rows, buttons

    rows_of.orig = orig
    L.rows_of = rows_of
    return rows_of


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        raise SystemExit("usage: logsize_llm.py --size N [--subsets FILE] -- <llm_next_action.py arguments>")
    k = argv.index("--")
    own, rest = argv[:k], argv[k + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True, help="64, 256, 1024, 4096 or whole")
    ap.add_argument("--subsets", default=os.path.join(H, "runs/logsize/subsets.json"))
    a = ap.parse_args(own)
    if any(x.split("=")[0] in REFUSED for x in rest):
        raise SystemExit("refused: logsize_llm.py is dev only")
    pa = argparse.ArgumentParser(add_help=False)
    pa.add_argument("--dataset", required=True)
    pa.add_argument("--out", required=True)
    b, _ = pa.parse_known_args(rest)
    with open(a.subsets, "rb") as h:
        raw = h.read()
    keep = S.keep_ids(json.loads(raw), b.dataset, a.size)
    rows_of = install(keep)
    tr, _ = rows_of(b.dataset, "train")
    os.makedirs(b.out, exist_ok=True)
    info = {"plan": "PLAN_logsize_dev.md", "dataset": b.dataset, "size": a.size,
            "n_dialogues": len({r["convo_id"] for r in tr}), "n_train_rows": len(tr),
            "subsets": a.subsets, "subsets_sha256": hashlib.sha256(raw).hexdigest(),
            "argv": [sys.argv[0]] + argv, "utc_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(os.path.join(b.out, "pool.json"), "w") as h:
        json.dump(info, h, indent=1)
    print("pool", json.dumps({k: info[k] for k in ("dataset", "size", "n_dialogues", "n_train_rows")}), flush=True)
    return L.main(rest)


if __name__ == "__main__":
    sys.exit(main())
