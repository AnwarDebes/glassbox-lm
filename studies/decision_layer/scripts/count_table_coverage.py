#!/usr/bin/env python
"""e72 step 1 (train and dev only; CPU; read only): the share of dialogue tokens that get no vector in the C4-word table.

The table is the frozen export c4words_S1000_v1 of the text-understanding line (project notes, not published; copied on
2026-10-19 to data/tables/c4words_S1000_v1; its MANIFEST.sha256 is checked here). The tokenizer is the export's
tokenize_ref.py, loaded by path from the copy. Counted per example row (what the bits see), per dataset (ABCD,
STAR), split (train, dev) and field:
- cust: every customer utterance so far (view B's xcust);
- recent: the last 3 utterances with speaker tags (view B's xrecent);
- crecent: the customer's last 3 utterances (view C's xcrecent; dl.abcd.customer_recent).
An empty text is replaced by "(nothing yet)", as src/dl/static_text.py does. Also checked on every text: a port of
the rule (the one src/dl/static_text.py is to use) gives the reference's tokens and row ids.

    count_table_coverage.py      # writes results/explore/count_table_coverage.json (refuses to overwrite)
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dl.abcd import customer_recent, load_examples  # noqa: E402

TDIR = ROOT / "data" / "tables" / "c4words_S1000_v1"
OUT = ROOT / "results" / "explore" / "count_table_coverage.json"
EMPTY = "(nothing yet)"
# the port of tokenize_ref.py's rule
CONTR = [(re.compile(r"n't\b"), " n't"), (re.compile(r"'(s|re|ve|ll|d|m)\b"), r" '\1")]
TOK = re.compile(r"n't|'[a-z]+|[a-z0-9]+|[^\sa-z0-9]")


def port_tokens(text: str) -> list[str]:
    t = text.lower()
    for pat, rep in CONTR:
        t = pat.sub(rep, t)
    return TOK.findall(t)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def kind(w: str) -> str:
    if any(c.isdigit() for c in w):
        return "has a digit"
    if re.fullmatch(r"[a-z']+", w):
        return "letters"
    return "other"


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; not overwriting")
    t0 = time.time()
    man = {}
    for line in (TDIR / "MANIFEST.sha256").read_text().splitlines():
        h, name = line.split()
        man[name] = h
        got = sha(TDIR / name)
        if got != h:
            raise SystemExit(f"{name}: sha256 {got} != {h}")
    spec = importlib.util.spec_from_file_location("tokenize_ref_c4words", TDIR / "tokenize_ref.py")
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    row = ref.rows()
    out = {"what": "e72 step 1: tokens without a vector in the C4-word table (train/dev only)", "table": "c4words_S1000_v1",
           "manifest": man, "n_words": len(row), "per": {}}
    n_checked = n_mismatch = 0
    for ds in ("abcd", "star"):
        for split in ("train", "dev"):
            rows, _ = load_examples(split, ds)
            fields = {"cust": [r["cust"] for r in rows], "recent": [r["recent"] for r in rows],
                      "crecent": [customer_recent(r) for r in rows]}
            for fld, texts in fields.items():
                n_tok = n_unk = n_none = n_any = 0
                unk, unk_kind = Counter(), Counter()
                types, unk_types = set(), set()
                for t in texts:
                    t = t or EMPTY
                    tk = ref.tokens(t)
                    pt = port_tokens(t)
                    n_checked += 1
                    if pt != tk or [row[w] for w in pt if w in row] != ref.ids(t):
                        n_mismatch += 1
                    u = [w for w in tk if w not in row]
                    n_tok += len(tk)
                    n_unk += len(u)
                    n_none += len(u) == len(tk)
                    n_any += bool(u)
                    unk.update(u)
                    unk_kind.update(kind(w) for w in u)
                    types.update(tk)
                    unk_types.update(u)
                out["per"][f"{ds}/{split}/{fld}"] = {
                    "n_texts": len(texts), "n_tokens": n_tok, "n_tokens_no_vector": n_unk,
                    "share_tokens_no_vector": round(n_unk / max(n_tok, 1), 6),
                    "share_texts_with_a_dropped_token": round(n_any / len(texts), 6),
                    "n_texts_no_known_token": n_none, "n_types": len(types), "n_types_no_vector": len(unk_types),
                    "dropped_by_kind": dict(unk_kind), "top_dropped": unk.most_common(40)}
                r = out["per"][f"{ds}/{split}/{fld}"]
                print(f"{ds:4} {split:5} {fld:7} tokens {n_tok:>9,}  no vector {100 * r['share_tokens_no_vector']:.2f}%  "
                      f"texts with a drop {100 * r['share_texts_with_a_dropped_token']:.1f}%  texts with no known token {n_none}  "
                      f"top: {', '.join(w for w, _ in unk.most_common(8))}", flush=True)
    out["port_check"] = {"texts": n_checked, "mismatches": n_mismatch}
    out["seconds"] = round(time.time() - t0, 1)
    out["time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"port check: {n_mismatch} mismatches on {n_checked:,} texts; {out['seconds']} s; wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
