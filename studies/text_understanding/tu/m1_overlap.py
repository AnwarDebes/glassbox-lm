#!/usr/bin/env python
"""M1 step 0 (tu/PLAN_M1.md "Step 0"; CPU only, before any model is scored): overlap of the TEST texts with the texts
our tables and readers were fit on. It reads test texts, never test labels.

    python -I -B m1_overlap.py --out tu/m1/overlap_m1.json

Keys of a text (as tu/c3x_rk.py):
- exact: build_corpus_c4.norm (NFKC, lowercased, its words joined by one space);
- order-free: the sorted words of that norm;
- reader: the frozenset of wordtab_feats.glove_tokens that have a live row in the a3B table (two texts with the same
  set get identical existsT bits under that table).
Counted per set, against:
- 3a's fit sentences (tu/a3/c4_sent_train_f.txt) and held-out sentences (c4_sent_held_f.txt): exact and order-free.
  Exact should be 0 by construction (a3_sentences.py drops evaluation keys);
- 3c's kept_F paraphrases of the SAME set (tu/c3/kept_F.jsonl): exact, order-free and reader;
- a3_teacher.py's 64 check sentences: exact, against every evaluation text of build_corpus_c4.eval_texts().
"matched_test_idx" per set is the union of the test indices hit by any key above except the check sentences (which
never entered a fit). Nothing is removed (PLAN_M1's rule); the verdict reports each arm's accuracy without them.
Refuses to overwrite.
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EMB = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EMB / "w1"))
sys.path.insert(0, f"{_REPO_ROOT}/studies/decision_layer/src")
A3 = HERE / "a3"
TABLE = A3 / "tables" / "a3B_d768.npz"
SETS = ("clinc150", "banking77", "hwu64")


def sha16(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def keys(text: str, live: set, norm, glove_tokens) -> tuple[str, tuple, frozenset]:
    n = norm(text)
    return n, tuple(sorted(n.split())), frozenset(w for w in glove_tokens(text) if w in live)


def overlap(test_texts: list[str], pool_keys: dict[str, set], live: set, norm, glove_tokens, examples: int = 5) -> dict:
    """Counts of test texts whose exact / order-free / reader key is in the pool's key sets (empty keys never match)."""
    hit = {k: [] for k in pool_keys}
    for i, t in enumerate(test_texts):
        e, o, r = keys(t, live, norm, glove_tokens)
        for name, kset in pool_keys.items():
            kk = {"exact": e, "order_free": o, "reader": r}[name.split(":")[-1]]
            if kk and kk in kset:
                hit[name].append(i)
    return {"counts": {k: len(v) for k, v in hit.items()}, "idx": {k: v for k, v in hit.items()},
            "examples": {k: [test_texts[i] for i in v[:examples]] for k, v in hit.items()}}


def c4_hits(paths: list[Path], tests: dict[str, list[str]], norm) -> tuple[dict, int]:
    """One pass over the C4 sentence files: per set, the test indices whose exact or order-free key equals a sentence's.
    Only the test keys are held in memory (not the 2M sentence keys)."""
    maps = {}
    for d, texts in tests.items():
        em, om = {}, {}
        for i, t in enumerate(texts):
            e = norm(t)
            if e:
                em.setdefault(e, []).append(i)
                om.setdefault(tuple(sorted(e.split())), []).append(i)
        maps[d] = (em, om)
    hits = {d: {"c4:exact": set(), "c4:order_free": set()} for d in tests}
    n = 0
    for p in paths:
        with open(p) as f:
            for line in f:
                n += 1
                e = norm(line.rstrip("\n"))
                if not e:
                    continue
                o = tuple(sorted(e.split()))
                for d, (em, om) in maps.items():
                    if e in em:
                        hits[d]["c4:exact"].update(em[e])
                    if o in om:
                        hits[d]["c4:order_free"].update(om[o])
    return hits, n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--c4", type=Path, nargs="+", default=[A3 / "c4_sent_train_f.txt", A3 / "c4_sent_held_f.txt"])
    ap.add_argument("--kept", type=Path, default=HERE / "c3" / "kept_F.jsonl")
    ap.add_argument("--load", default="", help="tests only: FILE.py:function returning a stand-in dataset")
    ap.add_argument("--skip-check-sentences", action="store_true", help="tests only: no evaluation texts are read")
    a = ap.parse_args()
    if a.out.exists():
        print(f"refusing to overwrite {a.out}", file=sys.stderr)
        return 2
    import a3_teacher
    from c3_common import B  # imports build_corpus_c4, which hides the GPU (a CPU job)
    from m1_feats import load_ds
    from wordtab_feats import glove_tokens

    t0 = time.time()
    z = np.load(TABLE)
    live = {str(w) for w, ok in zip(z["words"], np.abs(z["codes"].astype(np.float32)).sum(1) > 0) if ok}
    tests = {d: list(load_ds(d, a.load).texts["test"]) for d in SETS}
    hits, n_c4 = c4_hits(a.c4, tests, B.norm)
    kept = [json.loads(x) for x in a.kept.read_text().splitlines()]
    out = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "c4_files": {str(p): sha16(p) for p in a.c4},
           "c4_sentences": n_c4, "kept": str(a.kept), "kept_sha16": sha16(a.kept), "table_sha16": sha16(TABLE),
           "m1_overlap.py": sha16(Path(__file__).resolve()), "sets": {}}
    for d in SETS:
        test = tests[d]
        ke = {"kept:exact": set(), "kept:order_free": set(), "kept:reader": set()}
        for c in kept:
            if c["set"] == d:
                e, o, r = keys(c["text"], live, B.norm, glove_tokens)
                ke["kept:exact"].add(e)
                ke["kept:order_free"].add(o)
                ke["kept:reader"].add(r)
        res = overlap(test, ke, live, B.norm, glove_tokens)
        idx = {**{k: sorted(v) for k, v in hits[d].items()}, **res["idx"]}
        union = sorted({i for v in idx.values() for i in v})
        out["sets"][d] = {"n_test": len(test), "counts": {k: len(v) for k, v in idx.items()}, "matched_test_idx": union,
                          "n_matched": len(union), "idx_by_key": idx,
                          "examples": {k: [test[i] for i in v[:5]] for k, v in idx.items()}}
    if a.skip_check_sentences:
        out["a3_teacher_check_sentences_in_eval"] = "not checked (--skip-check-sentences, tests only)"
    else:
        ev, _ = B.eval_texts()
        evk = {}
        for name, texts in ev.items():
            for t in texts:
                evk.setdefault(B.norm(t), set()).add(name)
        out["a3_teacher_check_sentences_in_eval"] = {s: sorted(evk[B.norm(s)]) for s in a3_teacher.CHECK if B.norm(s) in evk}
        out["a3_teacher.py"] = sha16(Path(a3_teacher.__file__))
    out["seconds"] = round(time.time() - t0, 1)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "x") as f:
        f.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps({d: {"n_test": v["n_test"], "n_matched": v["n_matched"], **v["counts"]} for d, v in out["sets"].items()}))
    print(json.dumps({"a3_teacher_check_sentences_in_eval": out["a3_teacher_check_sentences_in_eval"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
