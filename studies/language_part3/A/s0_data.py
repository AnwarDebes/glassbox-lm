#!/usr/bin/env python
"""S0 (CPU): MultiWOZ 2.2 train replies -> A/out/s0/ (corpus.npz, meta.json, replies_delex.txt, bit_list.json).

    chain/part3_cpu_job.sh A tm s0_data.py [--out DIR] [--mwoz DIR] [--c4 DIR]

Rules: astruct/data.py (registered in REGISTRATION_R1.md). Opens only train/*.json, dialog_acts.json and
MANIFEST.sha256 of the MultiWOZ folder (asserted), and the C4 table's words.txt, vectors.npy and tokenize_ref.py.
Self-checks on every reply: the span_info test (review 2) and the token-count test (review 17).
The frozen bit list (bit_list.json) and its SHA-256 are written here; that SHA-256 goes into LOG.md before S2a.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from astruct import paths
from astruct.data import Corpus, make_corpus
from astruct.rows import bit_names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mwoz", default=str(paths.MWOZ))
    ap.add_argument("--c4", default=str(paths.C4))
    ap.add_argument("--out", default=str(paths.OUT / "s0"))
    args = ap.parse_args()
    env = paths.check_env()
    out = paths.OutDir(Path(args.out))
    log = paths.Log(out / "log.txt")
    log("S0 start; env", json.dumps(env))
    tk = paths.load_tokenize_ref()
    opened = []

    def opener(p, *a, **k):
        opened.append(str(p))
        return open(p, *a, **k)

    mw = Path(args.mwoz)
    arrays, meta = make_corpus(mw, Path(args.c4), tk.tokens, log=log, opener=opener)
    rel = [str(Path(p).relative_to(mw)) for p in opened]
    bad = [p for p in rel if not (p.startswith("train/") or p == "dialog_acts.json")]
    if bad:
        raise SystemExit(f"S0 opened files outside train/ and dialog_acts.json: {bad}")
    meta["opened_json"] = rel
    c = Corpus(arrays, meta)
    c.save(out.part)
    with open(out / "replies_delex.txt", "w") as f:
        f.writelines(" ".join(c.vocab[w] for w in c.tokens(r)) + "\n" for r in range(c.n))
    names, cd, n_pos_bits = bit_names(c)
    bl = json.dumps({"names": names, "count_derived": cd.tolist(), "n_position_bits": n_pos_bits})
    (out / "bit_list.json").write_text(bl)
    bl_sha = hashlib.sha256(bl.encode()).hexdigest()
    summary = {"counts": meta["counts"], "checks": meta["checks"], "source": meta["source"],
               "bits": len(names), "count_derived_bits": int(cd.sum()), "bits_A_nf": int((~cd).sum()),
               "bit_list_sha256": bl_sha, "code_sha256": paths.code_hashes(), "env": env, "utc_end": paths.utc()}
    log("S0 counts", json.dumps(meta["counts"]))
    log(f"bit list: {len(names)} bits, {int(cd.sum())} count-derived, A-nf {int((~cd).sum())}; bit_list.json sha256 {bl_sha}")
    log("NEXT: write this bit-list SHA-256 into LOG.md before S2a (s2a_rows.py refuses to start without it)")
    out.done(summary)


if __name__ == "__main__":
    main()
