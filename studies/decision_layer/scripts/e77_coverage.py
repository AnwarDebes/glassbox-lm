#!/usr/bin/env python
"""e77 (plan logs/plan_e77_multiwoz_dev.md; train and dev only; CPU; read only): count_table_coverage.py's count on
MultiWOZ: the share of tokens that get no vector in the C4-word table, per split and field.

count_table_coverage.py is imported unedited (its helpers, its tokenizer check and its table checks); only its loop over
datasets is repeated here, for "multiwoz". Writes results/explore/count_table_coverage_multiwoz.json (refuses to
overwrite). Added 2026-10-19.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import count_table_coverage as ctc  # noqa: E402

OUT = ROOT / "results" / "explore" / "count_table_coverage_multiwoz.json"


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; not overwriting")
    t0 = time.time()
    man = {}
    for line in (ctc.TDIR / "MANIFEST.sha256").read_text().splitlines():
        h, name = line.split()
        man[name] = h
        if ctc.sha(ctc.TDIR / name) != h:
            raise SystemExit(f"{name}: sha256 mismatch")
    spec = importlib.util.spec_from_file_location("tokenize_ref_c4words", ctc.TDIR / "tokenize_ref.py")
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    row = ref.rows()
    out = {"what": "e77: tokens without a vector in the C4-word table, MultiWOZ train/dev", "table": "c4words_S1000_v1",
           "manifest": man, "n_words": len(row), "per": {}}
    n_checked = n_mismatch = 0
    for split in ("train", "dev"):
        rows, _ = ctc.load_examples(split, "multiwoz")
        fields = {"cust": [r["cust"] for r in rows], "recent": [r["recent"] for r in rows]}
        for fld, texts in fields.items():
            n_tok = n_unk = n_none = n_any = 0
            unk, unk_kind = Counter(), Counter()
            types, unk_types = set(), set()
            for t in texts:
                t = t or ctc.EMPTY
                tk = ref.tokens(t)
                pt = ctc.port_tokens(t)
                n_checked += 1
                if pt != tk or [row[w] for w in pt if w in row] != ref.ids(t):
                    n_mismatch += 1
                u = [w for w in tk if w not in row]
                n_tok += len(tk)
                n_unk += len(u)
                n_none += len(u) == len(tk)
                n_any += bool(u)
                unk.update(u)
                unk_kind.update(ctc.kind(w) for w in u)
                types.update(tk)
                unk_types.update(u)
            r = out["per"][f"multiwoz/{split}/{fld}"] = {
                "n_texts": len(texts), "n_tokens": n_tok, "n_tokens_no_vector": n_unk,
                "share_tokens_no_vector": round(n_unk / max(n_tok, 1), 6),
                "share_texts_with_a_dropped_token": round(n_any / len(texts), 6),
                "n_texts_no_known_token": n_none, "n_types": len(types), "n_types_no_vector": len(unk_types),
                "dropped_by_kind": dict(unk_kind), "top_dropped": unk.most_common(40)}
            print(f"multiwoz {split:5} {fld:7} tokens {n_tok:>10,}  no vector {100 * r['share_tokens_no_vector']:.2f}%  "
                  f"texts with a drop {100 * r['share_texts_with_a_dropped_token']:.1f}%  texts with no known token {n_none}  "
                  f"top: {', '.join(w for w, _ in unk.most_common(10))}", flush=True)
    out["port_check"] = {"texts": n_checked, "mismatches": n_mismatch}
    out["seconds"] = round(time.time() - t0, 1)
    out["time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"port check: {n_mismatch} mismatches on {n_checked:,} texts; {out['seconds']} s; wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
