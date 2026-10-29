#!/usr/bin/env python
"""Fresh intent test set: freezes the list of MTOP English TEST items that the secondary analysis leaves out (2026-10-23: primary on all 4,386 test items; secondary, with no rule, without the items exact or within one word of
anything our pipelines trained on). Reads fresh/mtop_overlap.json and the ID column (column 1) of en/test.txt; no label
column is read.

    python -I -B fresh/mtop_freeze_matched.py --zip data_untrusted/mtop/s3v_qLlhqQW2ypnQQXRE/mtop.zip \
        --overlap fresh/mtop_overlap.json --out fresh/mtop_test_matched_v1.json
    python -I -B fresh/mtop_freeze_matched.py --selftest

Left out: a test item whose exact, order-free or one-edit key matches a text of a TRAINED-ON pool: ours_train_val
(CLINC150, Banking77, HWU64 and MASSIVE en train and val), paraphrases (3c's generated, kept and EDA texts) and c4_fit
(3a's C4 fit and held-out sentences). The pools ours_test_texts and other_eval (evaluated on, never trained on) and
mtop_internal are reported, not used. The 3c paraphrase exact matches get their own list (Qwen3.5-9B may have seen
MTOP). Positions are 0-based over the non-empty rows of en/test.txt, the order fresh/mtop_overlap.py used.
Refuses to overwrite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

TRAINED = ("ours_train_val", "paraphrases", "c4_fit")
KEYS = ("exact", "order_free", "one_edit")


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def matched(hits: dict, pools=TRAINED, keys=KEYS) -> list[int]:
    return sorted({i for p in pools for k in keys for i in hits[p][k]})


def selftest() -> None:
    h = {"ours_train_val": {"exact": [1], "order_free": [1, 4], "one_edit": [1, 7]},
         "paraphrases": {"exact": [2], "order_free": [], "one_edit": [2, 9]},
         "c4_fit": {"exact": [], "order_free": [], "one_edit": [3]},
         "ours_test_texts": {"exact": [5], "order_free": [5], "one_edit": [5, 6]}}
    assert matched(h) == [1, 2, 3, 4, 7, 9], matched(h)
    assert 5 not in matched(h) and 6 not in matched(h), "an evaluated-on pool must not be used"
    print("SELFTEST PASSED")


def main() -> None:
    if sys.argv[1:] == ["--selftest"]:
        selftest()
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, required=True)
    ap.add_argument("--overlap", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists(), f"{args.out} exists"
    ov = json.loads(args.overlap.read_text())
    assert ov["files"]["mtop.zip"] == sha256_file(args.zip), "the overlap file was made from another zip"
    rows = [ln.split("\t") for ln in zipfile.ZipFile(args.zip).read("mtop/en/test.txt").decode("utf-8").split("\n") if ln.strip()]
    assert len(rows) == ov["mtop_sizes"]["test"] and all(len(r) == 8 for r in rows)
    ids = [r[0] for r in rows]  # column 1, the utterance ID; no other column is read
    assert len(set(ids)) == len(ids), "the test IDs are not unique"
    h = ov["hits"]["test"]
    out_pos = matched(h)
    para_exact = sorted(h["paraphrases"]["exact"])
    res = {"rule": "left out of the secondary analysis: exact, order-free or one-edit match with a trained-on pool",
           "trained_on_pools": list(TRAINED), "keys": list(KEYS), "n_test": len(rows), "n_left_out": len(out_pos),
           "n_kept": len(rows) - len(out_pos), "positions": out_pos, "ids": [ids[i] for i in out_pos],
           "per_pool_union": {p: len(matched(h, (p,))) for p in TRAINED},
           "not_used_pools": {p: len(matched(h, (p,))) for p in ("ours_test_texts", "other_eval", "mtop_internal")},
           "paraphrase_exact": {"n": len(para_exact), "positions": para_exact, "ids": [ids[i] for i in para_exact],
                                "note": "3c's Qwen3.5-9B paraphrases equal these MTOP test texts; Qwen may have seen MTOP"},
           "sources": {"mtop.zip": sha256_file(args.zip), "mtop_overlap.json": sha256_file(args.overlap),
                       "mtop_freeze_matched.py": sha256_file(Path(__file__).resolve())},
           "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    args.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("positions", "ids")} | {"paraphrase_exact": res["paraphrase_exact"]["n"]}))


if __name__ == "__main__":
    main()
