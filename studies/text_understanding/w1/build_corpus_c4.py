#!/usr/bin/env python
"""W1 (a Tsetlin machine learns its own word table): the nested, seeded C4 slices (PLAN_tm_word_table.md, "Corpus").

    python -I build_corpus_c4.py --shards 00000,00001            # writes w1/corpus/ (refuses to overwrite)
    python -I build_corpus_c4.py --shards 00000 --smoke 3000     # first 3,000 documents of the order, nothing kept

Input:   data_untrusted/c4_en/c4-train.{shard}-of-01024.json.gz (allenai/c4, en, at the revision in SOURCE.txt), read
         as plain JSON lines; only the "text" field is used. The raw text that potion-base-32M's Tokenlearn step was
         trained on (its model card names minishlab/tokenlearn-c4-en-bge-base-v1.5).
Order:   all documents of the given shards, permuted once by numpy default_rng(2026).
Filter:  sentence level, before any tokenization. A document is split into lines and each line into sentences
         (after . ! or ? followed by white space). A sentence is dropped when
         (a) its key equals the key of any evaluation text (the test 5 rule: sha256 of the NFKC + lowercased text
             reduced to its words [a-z0-9]+('[a-z0-9]+)* joined by one space, curly apostrophes mapped to '), or
         (b) it holds 8 consecutive words that also occur, in that order, in an evaluation text of 8 or more words.
         Evaluation texts (every split we evaluate on, plus N7's published train sets, which hold N7's carved val):
         CLINC150 val/test/oos_val/oos_test; clinc_oos_plus val/test/oos_val/oos_test; Banking77, HWU64,
         MASSIVE en-US and nb-NO val/test (dl.data); TREC-6, MR, SST-2, R52, R8: published train (+ val.csv), SST-2
         dev, test (data_untrusted, N7); ABCD and STAR dev/test (the decision-layer study's examples files: every
         turn of the cust, recent and full fields). Only counts and hashes are recorded; no text is printed or kept
         outside the token arrays.
Tokens:  potion-base-32M tokenizer (revision 1e5a03f8, tokenizer.json of that snapshot), add_special_tokens=False, as
         every existsT run. [UNK] stays in the stream (it gets no vector and is no context; counted).
Slices:  S1, S10, S100 = the shortest prefixes of the filtered order holding at least 1M / 10M / 100M tokens, so each
         smaller slice is nested in the larger ones. H = documents taken from the END of the order until at least
         2M tokens, disjoint from S100: the held-out text for the Tsetlin machine's own self-supervised check (it is
         never an evaluation set and never used for a choice that an evaluation set should make).
Output:  corpus/c4_S100.ids.npy and corpus/c4_H.ids.npy (int32 token ids, documents separated by -1),
         corpus/c4_slices.json (token and document counts per slice, end offsets of S1 / S10 inside S100, sha256 of
         each slice's int32 bytes, input and script hashes, filter counts per evaluation split).
CPU only; 2 tokenizer threads.
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))
_HOME = _os.path.expanduser("~")

import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["RAYON_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse  # noqa: E402
import csv  # noqa: E402
import gzip  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import unicodedata  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent  # .../studies/text_understanding/w1
EMB = HERE.parent  # .../studies/text_understanding
U = EMB / "data_untrusted"
C4 = U / "c4_en"
OUT = HERE / "corpus"
DLROOT = Path(f"{_REPO_ROOT}/studies/decision_layer")
sys.path.insert(0, str(DLROOT / "src"))
from dl.data import load  # noqa: E402

REV = "1e5a03f8eeb2c98b928fbbd846f22f816360919f"
TOK_JSON = Path(f"{_HOME}/.cache/huggingface/hub/models--minishlab--potion-base-32M/snapshots") / REV / "tokenizer.json"
OOS_PLUS = U / "clinc_oos_plus" / "data_oos_plus.json"
WORD = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*")
SENT = re.compile(r"(?<=[.!?])\s+")
TURN_TAG = re.compile(r"^\s*(agent|customer|user|system|wizard)\s*:\s*", re.I)
SIZES = {"S1": 1_000_000, "S10": 10_000_000, "S100": 100_000_000}
H_TOKENS = 2_000_000
NGRAM = 8
SEED = 2026


def norm(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).replace("’", "'").replace("‘", "'").lower()
    return " ".join(WORD.findall(t))


def key(text: str) -> str:
    return hashlib.sha256(norm(text).encode("utf-8")).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_ids(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.int32).tobytes()).hexdigest()


def read_csv_texts(p: Path) -> list[str]:
    csv.field_size_limit(10**9)
    with open(p, newline="", encoding="utf-8") as f:
        return [r["text"] for r in csv.DictReader(f)]


def read_jsonl_texts(p: Path) -> list[str]:
    return [json.loads(x)["text"] for x in open(p, encoding="utf-8") if x.strip()]


def eval_texts() -> tuple[dict[str, list[str]], dict[str, str]]:
    """Every evaluation text, by split name, and the sha256 of each file read."""
    out: dict[str, list[str]] = {}
    files: dict[str, str] = {}
    ds = {n: load(n) for n in ("clinc150", "banking77", "hwu64", "massive_en", "massive_nb")}
    for n, d in ds.items():
        files[f"dl.data:{n}"] = json.dumps(d.sources, sort_keys=True)
    oos_plus = json.loads(OOS_PLUS.read_text())
    files[str(OOS_PLUS.relative_to(EMB))] = sha256_file(OOS_PLUS)
    for sp in ("val", "test", "oos_val", "oos_test"):
        out[f"clinc150/{sp}"] = list(ds["clinc150"].texts[sp])
        out[f"clinc_oos_plus/{sp}"] = [t for t, _ in oos_plus[sp]]
    for n in ("banking77", "hwu64", "massive_en", "massive_nb"):
        for sp in ("val", "test"):
            out[f"{n}/{sp}"] = list(ds[n].texts[sp])
    n7 = {"trec/train": U / "trec_setfit" / "train.jsonl", "trec/test": U / "trec_setfit" / "test.jsonl",
          "sst2/train": U / "sst2_setfit" / "train.jsonl", "sst2/dev": U / "sst2_setfit" / "dev.jsonl",
          "sst2/test": U / "sst2_setfit" / "test.jsonl"}
    for name, p in n7.items():
        out[name] = read_jsonl_texts(p)
        files[str(p.relative_to(EMB))] = sha256_file(p)
    for ds_name, folder in (("mr", "mr_dxgp"), ("r52", "r52_dxgp"), ("r8", "r8_dxgp")):
        for sp in ("train", "val", "test"):
            p = U / folder / f"{sp}.csv"
            out[f"{ds_name}/{sp}"] = read_csv_texts(p)
            files[str(p.relative_to(EMB))] = sha256_file(p)
    for dn in ("abcd", "star"):
        for sp in ("dev", "test"):
            p = DLROOT / "data" / dn / f"examples_{sp}.json.gz"
            rows = json.load(gzip.open(p))["rows"]
            turns = []
            for r in rows:
                for fld in ("cust", "recent", "full"):
                    for t in str(r.get(fld) or "").split(" | "):
                        t = TURN_TAG.sub("", t).strip()
                        if t:
                            turns.append(t)
            out[f"{dn}/{sp}"] = turns
            files[str(p)] = sha256_file(p)
    return out, files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True, help="comma-separated shard numbers, e.g. 00000,00001")
    ap.add_argument("--smoke", type=int, default=0, help="only the first N documents of the order; nothing kept")
    ap.add_argument("--chunk", type=int, default=2000)
    args = ap.parse_args()
    t0 = time.time()
    if not args.smoke:
        OUT.mkdir(parents=True, exist_ok=True)
        for f in ("c4_S100.ids.npy", "c4_H.ids.npy", "c4_slices.json"):
            assert not (OUT / f).exists(), f"{OUT / f} exists: a new version needs a new name"
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(TOK_JSON))
    unk = tok.token_to_id("[UNK]")

    # ---- blocklist ----
    ev, ev_files = eval_texts()
    empty = key("")
    block, grams, block_info = {}, {}, {}
    for name, texts in ev.items():
        ks = {key(t) for t in texts}
        ks.discard(empty)
        block[name] = ks
        g = set()
        for t in texts:
            w = norm(t).split()
            for i in range(len(w) - NGRAM + 1):
                g.add(" ".join(w[i : i + NGRAM]))
        grams[name] = g
        block_info[name] = {"texts": len(texts), "distinct_keys": len(ks), "distinct_8grams": len(g)}
    union_keys = set().union(*block.values())
    union_grams = set().union(*grams.values())
    print(f"blocklist: {len(ev)} splits, {len(union_keys):,} keys, {len(union_grams):,} 8-grams ({time.time() - t0:.0f}s)", flush=True)

    # ---- documents ----
    shards = args.shards.split(",")
    paths = [C4 / f"c4-train.{s}-of-01024.json.gz" for s in shards]
    docs: list[str] = []
    for p in paths:
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                docs.append(json.loads(line)["text"])
    order = np.random.default_rng(SEED).permutation(len(docs))
    print(f"documents: {len(docs):,} from {len(paths)} shard(s) ({time.time() - t0:.0f}s)", flush=True)

    drop = {"sentences": 0, "kept_sentences": 0, "key": 0, "ngram": 0}
    per_split = {name: {"key": 0, "ngram": 0} for name in ev}

    def clean(doc: str) -> str:
        lines = []
        for line in doc.split("\n"):
            kept = []
            for s in SENT.split(line):
                if not s.strip():
                    continue
                drop["sentences"] += 1
                k = key(s)
                if k in union_keys:
                    drop["key"] += 1
                    for name, ks in block.items():
                        if k in ks:
                            per_split[name]["key"] += 1
                    continue
                w = norm(s).split()
                hit = [" ".join(w[i : i + NGRAM]) for i in range(len(w) - NGRAM + 1)]
                hit = [g for g in hit if g in union_grams]
                if hit:
                    drop["ngram"] += 1
                    for name, g in grams.items():
                        if any(h in g for h in hit):
                            per_split[name]["ngram"] += 1
                    continue
                kept.append(s)
                drop["kept_sentences"] += 1
            if kept:
                lines.append(" ".join(kept))
        return "\n".join(lines)

    def take(idx_iter, need: int | None):
        """Documents in the given order until `need` tokens (need None: until every size in SIZES is reached).
        Returns the int32 stream (-1 after each document), tokens, documents kept, documents left empty by the
        filter, the end (offset, documents, tokens) of each size, and every document index looked at."""
        st = {"parts": [], "n_tok": 0, "n_docs": 0, "n_empty": 0, "len": 0, "ends": {}, "used": []}
        targets = sorted(SIZES.items(), key=lambda kv: kv[1]) if need is None else []

        def add(res):
            for j, ids in res:
                st["used"].append(j)
                if len(ids) == 0:
                    st["n_empty"] += 1
                    continue
                st["parts"].append(np.asarray(ids, dtype=np.int32))
                st["parts"].append(np.array([-1], dtype=np.int32))
                st["len"] += len(ids) + 1
                st["n_tok"] += len(ids)
                st["n_docs"] += 1
                for name, size in targets:
                    if name not in st["ends"] and st["n_tok"] >= size:
                        st["ends"][name] = (st["len"], st["n_docs"], st["n_tok"])

        def done() -> bool:
            return len(st["ends"]) == len(SIZES) if need is None else st["n_tok"] >= need

        buf = []
        for i in idx_iter:
            buf.append(int(i))
            if len(buf) == args.chunk:
                add(encode(buf))
                buf = []
                if done():
                    break
        if buf and not done():
            add(encode(buf))
        arr = np.concatenate(st["parts"]) if st["parts"] else np.zeros(0, np.int32)
        return arr, st["n_tok"], st["n_docs"], st["n_empty"], st["ends"], st["used"]

    def encode(idx: list[int]):
        texts = [clean(docs[i]) for i in idx]
        enc = tok.encode_batch(texts, add_special_tokens=False)
        return [(i, e.ids) for i, e in zip(idx, enc)]

    if args.smoke:
        arr, n_tok, n_docs, n_empty, _, used = take(iter(order[: args.smoke]), need=10**15)
        print(json.dumps({"smoke_docs": len(used), "tokens": n_tok, "docs_kept": n_docs, "docs_empty": n_empty,
                          "tokens_per_doc": round(n_tok / max(n_docs, 1), 1), "unk_share": round(float((arr == unk).sum()) / max(n_tok, 1), 5),
                          "filter": drop, "per_split_nonzero": {k: v for k, v in per_split.items() if v["key"] or v["ngram"]},
                          "seconds": round(time.time() - t0, 1)}))
        return

    s100, n_tok, n_docs, n_empty, ends, used = take(iter(order), need=None)
    assert len(ends) == len(SIZES), f"the shards hold only {n_tok:,} tokens: add a shard"
    used_set = set(used)
    end100 = ends["S100"][0]
    s100 = s100[:end100]
    rest = [i for i in order[::-1] if int(i) not in used_set]
    h, h_tok, h_docs, h_empty, _, h_used = take(iter(rest), need=H_TOKENS)
    assert h_tok >= H_TOKENS and not (set(h_used) & used_set)
    np.save(OUT / "c4_S100.ids.npy", s100)
    np.save(OUT / "c4_H.ids.npy", h)
    slices = {}
    for name in sorted(SIZES, key=SIZES.get):
        end, nd, nt = ends[name]
        a = s100[:end]
        slices[name] = {"end_offset_in_S100": int(end), "documents": int(nd), "tokens": int(nt),
                        "unk_tokens": int((a == unk).sum()), "distinct_types": int(len(np.unique(a[a >= 0]))),
                        "sha256_int32": sha256_ids(a)}
    slices["H"] = {"documents": int(h_docs), "tokens": int(h_tok), "unk_tokens": int((h == unk).sum()),
                   "distinct_types": int(len(np.unique(h[h >= 0]))), "sha256_int32": sha256_ids(h)}
    man = {
        "written": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "inputs": {"shards": {p.name: sha256_file(p) for p in paths},
                   "tokenizer": {"revision": REV, "sha256": sha256_file(TOK_JSON)}, "eval_files": ev_files},
        "order": f"numpy default_rng({SEED}).permutation over {len(docs):,} documents",
        "filter": {"rule": "sentence dropped if its key equals an evaluation text's key, or it shares 8 consecutive "
                           "words with an evaluation text of 8 or more words", "counts": drop,
                   "per_eval_split": per_split, "blocklist": block_info,
                   "documents_empty_after_filter_in_S100_scan": int(n_empty)},
        "slices": slices,
        "files": {"c4_S100.ids.npy": sha256_file(OUT / "c4_S100.ids.npy"), "c4_H.ids.npy": sha256_file(OUT / "c4_H.ids.npy")},
        "seconds": round(time.time() - t0, 1),
    }
    (OUT / "c4_slices.json").write_text(json.dumps(man, indent=1))
    print(json.dumps({"slices": slices, "filter": drop, "seconds": man["seconds"]}))


if __name__ == "__main__":
    main()
