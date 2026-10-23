#!/usr/bin/env python
"""e82 self-test (CPU; train and dev only; no model): a new count table in the agent's view B, before any e82 run.

    e82_selftest.py --table-dir data/tables/<folder> --name <name> | tee logs/e82_selftest_<name>.out

Checks, stopping at the first failure:
1. the table registers (scripts/newtable_run.register: MANIFEST hash = file hash) and loads through
   dl.count_text.CountTable (its hash and tokenizer checks);
2. its word list against the C4 table's (c4words-S1000-v1): equal, or the difference is counted and reported;
3. view B (mem, words, xcust, xrecent; lex_k 1000, seed 0) builds on STAR, ABCD and MultiWOZ, train fit, dev bits;
4. the mem and words blocks are byte-identical to view B on the C4 table (they do not read the table), and the C4
   view's dev bits equal the tuned rows' (e74c and e77b), so the reference is the one the R4 rule compares with.
Reported: how many of xcust's and xrecent's dev bits differ from the C4 view's, and the new blocks' on-share.
The plumbing was tested on 2026-10-21 by registering the C4 table itself under a test name, where every bit must be
equal. Added 2026-10-21.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(1, str(ROOT / "scripts"))

from dl import count_text, static_text  # noqa: E402
from newtable_run import register  # noqa: E402

REF = {"star": "e74c_star_Bc4_c16000_s16", "abcd": "e74c_abcd_Bc4_c4000_s16", "multiwoz": "e77_mwoz_Bc4_c16000_s8"}
PARTS = ("mem", "words", "xcust", "xrecent")


def need(cond: bool, msg: str) -> None:
    print(("ok   " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        sys.exit(1)


def view_bits(name: str, ds: str):
    import functools

    from dl import abcd

    static_text.ExistsT = functools.partial(count_text.ExistsTC, table_name=name)
    tr, bl = abcd.load_examples("train", ds)
    dv, _ = abcd.load_examples("dev", ds)
    E = np.load(ROOT / "data" / "emb" / f"{ds}__none.npz")
    emb = {s: {f: E[f"{s}__{f}"] for f in ("cust", "recent", "full")} for s in ("train", "dev")}
    v = abcd.ABCDView(parts=PARTS, lex_k=1000, seed=0).fit(tr, emb["train"], bl)
    X = v.transform(dv, emb["dev"])
    n_mem = 2 * len(bl) + 6
    return X, n_mem, len(v.lex.names), v.xcust.n_bits, v.xrecent.n_bits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table-dir", required=True)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    sha = register(Path(args.table_dir), args.name)
    tb = count_text.CountTable(args.name)
    need(tb.file_sha256 == sha, f"{args.name}: loads through CountTable, table.npz {sha[:16]}, tokenizer checked")
    c4 = count_text.count_table("c4words-S1000-v1")
    same = tb.words == c4.words
    print(f"   word list: {len(tb.words):,} words; {'equal to' if same else 'differs from'} the C4 table's ({len(c4.words):,}); "
          f"{len(set(tb.words) - set(c4.words)):,} new, {len(set(c4.words) - set(tb.words)):,} gone", flush=True)
    static_text.table = count_text._no_potion
    static_text.StaticTable = count_text._no_potion
    rows = {d["tag"]: d for d in map(json.loads, (ROOT / "results" / "explore" / "abcd.jsonl").read_text().splitlines())
            if d.get("arm") == "tm" and d.get("tag") in REF.values()}
    for ds, tag in REF.items():
        Xc, n_mem, n_words, nc, nr = view_bits("c4words-S1000-v1", ds)
        h = hashlib.sha256(np.ascontiguousarray(Xc, dtype=np.uint8).tobytes()).hexdigest()
        need(h == rows[tag]["bits_sha256"]["dev"], f"{ds}: view B on the C4 table = {tag}'s dev bits ({h[:8]})")
        Xn, n_mem2, n_words2, nc2, nr2 = view_bits(args.name, ds)
        k = n_mem + n_words
        need(n_mem2 == n_mem and n_words2 == n_words and np.array_equal(Xn[:, :k], Xc[:, :k]),
             f"{ds}: mem ({n_mem}) and words ({n_words}) bits byte-identical to the C4 view's")
        if Xn.shape == Xc.shape:
            diff = float((Xn[:, k:] != Xc[:, k:]).mean())
        else:
            diff = float("nan")
        print(f"   {ds}: new table view B {Xn.shape[1]} bits (xcust {nc2}, xrecent {nr2}; C4: {nc}, {nr}); "
              f"text-bit difference share {diff:.4f}; on-share of the text bits {Xn[:, k:].mean():.3f}", flush=True)
    print("all checks passed", flush=True)


if __name__ == "__main__":
    main()
