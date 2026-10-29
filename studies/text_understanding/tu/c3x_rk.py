#!/usr/bin/env python
"""Step 3c follow-up (a), post hoc and descriptive: CLINC150's kept paraphrases with F1 extended by the reader's own
keys (the leakage check's L4). CPU only.

    python -I -B c3x_rk.py --kept tu/c3/kept_F.jsonl --out tu/c3x/kept_F_rk_clinc150.jsonl

F1 keys a text by its normalised word SEQUENCE, but the existsT reader sees only the SET of a text's in-table tokens.
So a reordered or token-set copy of an evaluation text passes F1 and is, for the reader, a copy. This script keeps a
CLINC150 kept_F row only when neither of its two order-free keys equals the same key of an evaluation text:
- the reader key: the frozenset of the text's wordtab_feats.glove_tokens that have a row in the a3B table (two texts
  with the same set get identical existsT bits);
- the order-free word key: the sorted words of build_corpus_c4.norm.
The evaluation texts are F1's: c3_filter.leak_index's sources (build_corpus_c4.eval_texts() and the MultiWOZ 2.2 dev and
test utterances). Rows of other sets are not written. The counts and up to 20 dropped examples go to the meta file.
Never overwrites.
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))

import argparse
import glob
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
TABLE = HERE / "a3" / "tables" / "a3B_d768.npz"


def sha16(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def eval_texts(B, MW: Path) -> list[str]:
    ev, _ = B.eval_texts()
    texts = [t for v in ev.values() for t in v]
    for split in ("dev", "test"):
        for f in sorted(glob.glob(str(MW / split / "dialogues_*.json"))):
            for dlg in json.loads(Path(f).read_text()):
                texts += [turn["utterance"] for turn in dlg["turns"]]
    return texts


def keys_of(text: str, live: set, B, glove_tokens) -> tuple[frozenset, tuple]:
    return frozenset(w for w in glove_tokens(text) if w in live), tuple(sorted(B.norm(text).split()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--kept", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--set", default="clinc150")
    a = ap.parse_args()
    meta_p = a.out.with_suffix(".meta.json")
    for p in (a.out, meta_p):
        if p.exists():
            print(f"refusing to overwrite {p}", file=sys.stderr)
            return 2
    import c3_filter as CF  # imports build_corpus_c4, which hides the GPU (a CPU job)
    from c3_common import B
    from wordtab_feats import glove_tokens

    t0 = time.time()
    z = np.load(TABLE)
    codes = z["codes"].astype(np.float32)
    live = {str(w) for w, ok in zip(z["words"], np.abs(codes).sum(1) > 0) if ok}
    rk, ok = set(), set()
    for t in eval_texts(B, CF.MW):
        r, o = keys_of(t, live, B, glove_tokens)
        if r:
            rk.add(r)
        if o:
            ok.add(o)
    kept, dropped = [], []
    n = 0
    for line in a.kept.read_text().splitlines():
        c = json.loads(line)
        if c["set"] != a.set:
            continue
        n += 1
        r, o = keys_of(c["text"], live, B, glove_tokens)
        why = "reader_key" if r and r in rk else ("order_free_key" if o and o in ok else None)
        (dropped if why else kept).append({**c, "why": why} if why else c)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "x") as f:
        f.write("".join(json.dumps(c) + "\n" for c in kept))
    meta = {"set": a.set, "kept_in": str(a.kept), "kept_in_sha16": sha16(a.kept), "rows_in": n, "rows_out": len(kept),
            "dropped": len(dropped), "dropped_reader_key": sum(d["why"] == "reader_key" for d in dropped),
            "dropped_order_free_key": sum(d["why"] == "order_free_key" for d in dropped),
            "eval_reader_keys": len(rk), "eval_order_free_keys": len(ok), "table_sha16": sha16(TABLE),
            "examples": [{"text": d["text"], "source": d["source"], "label": d["label"], "why": d["why"]} for d in dropped[:20]],
            "out_sha16": sha16(a.out), "c3x_rk.py": sha16(Path(__file__).resolve()), "seconds": round(time.time() - t0, 1),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(meta_p, "x") as f:
        f.write(json.dumps(meta, indent=1) + "\n")
    print(json.dumps({k: meta[k] for k in ("rows_in", "rows_out", "dropped", "dropped_reader_key", "dropped_order_free_key")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
