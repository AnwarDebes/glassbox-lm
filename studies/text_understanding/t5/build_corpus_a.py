#!/usr/bin/env python
"""Test 5: build corpus (a) for the TM-autoencoder timing pilot (DESIGN_NOTES.md Sec. 4.1-4.2).

    python -I build_corpus_a.py            # writes t5/corpus_a.npz and t5/corpus_a.manifest.json

Texts:  the train texts of CLINC150 (+ its oos_train), Banking77 (the study split) and HWU64. No MASSIVE text.
Filter: every corpus text whose normalized key matches any val or test text of any evaluation split is dropped:
        CLINC150 val/test/oos_val/oos_test, data_untrusted/clinc_oos_plus val/test/oos_val/oos_test,
        Banking77 val/test, HWU64 val/test, MASSIVE en-US dev/test and MASSIVE nb-NO dev/test.
        key = sha256 of NFKC + lowercased text reduced to its words [a-z0-9]+('[a-z0-9]+)* joined by one space
        (curly apostrophes mapped to '). Eval texts with an empty key are left out of the blocklist (counted).
        Only counts and hashes are recorded; no text is printed or stored.
Tokens: potion-base-32M tokenizer (revision 1e5a03f8eeb2c98b928fbbd846f22f816360919f, tokenizer.json of that
        snapshot, the file SentenceTransformer loads for this model), add_special_tokens=False, as in
        exists_tm.py. Each text becomes its set of token ids.
Vocab:  the token types with df >= 2 in the filtered corpus (df = number of texts containing the type).
Output: corpus_a.npz with the binary documents x vocabulary matrix as CSR parts (indptr, indices: int32;
        data omitted, all ones), shape, vocab_ids (potion token ids, sorted ascending), df, doc_source;
        corpus_a.manifest.json with input hashes, filter counts per split, sizes and checks.
Read-only on all inputs. CPU only, one thread.
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))
_HOME = _os.path.expanduser("~")

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["RAYON_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import unicodedata  # noqa: E402
from collections import Counter  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent  # .../studies/text_understanding/t5
EMB = HERE.parent  # .../studies/text_understanding
DL_SRC = Path(f"{_REPO_ROOT}/studies/decision_layer/src")
sys.path.insert(0, str(DL_SRC))
from dl.data import load  # noqa: E402

HF_ID, REV = "minishlab/potion-base-32M", "1e5a03f8eeb2c98b928fbbd846f22f816360919f"
TOK_JSON = Path(f"{_HOME}/.cache/huggingface/hub/models--minishlab--potion-base-32M/snapshots") / REV / "tokenizer.json"
OOS_PLUS = EMB / "data_untrusted" / "clinc_oos_plus" / "data_oos_plus.json"
STUDY_META = {n: EMB / "emb" / f"{n}__p32tok_K512.meta.json" for n in ("clinc150", "banking77", "hwu64")}
OUT_NPZ, OUT_MAN = HERE / "corpus_a.npz", HERE / "corpus_a.manifest.json"
WORD = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*")
PROBES = ["balance", "transfer", "alarm", "weather", "french", "card", "music", "flight", "cancel", "email"]
SOURCES = [("clinc150", "train"), ("clinc150", "oos_train"), ("banking77", "train"), ("hwu64", "train")]
SOURCE_CODE = {f"{d}/{s}": i for i, (d, s) in enumerate(SOURCES)}


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


def main() -> None:
    t0 = time.time()
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(TOK_JSON))

    ds = {n: load(n) for n in ("clinc150", "banking77", "hwu64", "massive_en", "massive_nb")}
    oos_plus = json.loads(OOS_PLUS.read_text())

    # ---- blocklist: every val / test split of every evaluation dataset ----
    eval_texts: dict[str, list[str]] = {}
    for sp in ("val", "test", "oos_val", "oos_test"):
        eval_texts[f"clinc150/{sp}"] = list(ds["clinc150"].texts[sp])
        eval_texts[f"clinc_oos_plus/{sp}"] = [t for t, _ in oos_plus[sp]]
    for n in ("banking77", "hwu64", "massive_en", "massive_nb"):
        for sp in ("val", "test"):  # massive: val = the file's dev partition
            eval_texts[f"{n}/{sp}"] = list(ds[n].texts[sp])
    empty_key = key("")
    block: dict[str, set[str]] = {}
    block_info = {}
    for name, texts in eval_texts.items():
        ks = [key(t) for t in texts]
        n_empty = sum(1 for k in ks if k == empty_key)
        block[name] = {k for k in ks if k != empty_key}
        block_info[name] = {"texts": len(texts), "distinct_keys": len(block[name]), "empty_key_texts": n_empty}
    union_block = set().union(*block.values())

    # ---- corpus (a) sources and the filter ----
    kept_texts, kept_src, kept_keys = [], [], []
    per_source = {}
    dropped_per_split = {name: {"corpus_texts_matching": 0, "distinct_keys_matching": 0} for name in block}
    matched_keys_per_split = {name: set() for name in block}
    for d, sp in SOURCES:
        texts = list(ds[d].texts[sp])
        n_drop = n_empty = 0
        for t in texts:
            k = key(t)
            if k == empty_key:
                n_empty += 1
            if k in union_block:
                n_drop += 1
                for name, s in block.items():
                    if k in s:
                        dropped_per_split[name]["corpus_texts_matching"] += 1
                        matched_keys_per_split[name].add(k)
                continue
            kept_texts.append(t)
            kept_src.append(SOURCE_CODE[f"{d}/{sp}"])
            kept_keys.append(k)
        per_source[f"{d}/{sp}"] = {"texts": len(texts), "dropped": n_drop, "kept": len(texts) - n_drop,
                                   "empty_key_texts": n_empty}
    for name in block:
        dropped_per_split[name]["distinct_keys_matching"] = len(matched_keys_per_split[name])

    # ---- tokenization (potion), sets of ids ----
    enc = tok.encode_batch(kept_texts, add_special_tokens=False)
    doc_sets = [sorted(set(e.ids)) for e in enc]
    df = Counter()
    for s in doc_sets:
        df.update(s)
    vocab_ids = np.array(sorted(t for t, c in df.items() if c >= 2), dtype=np.int32)
    col = {int(t): i for i, t in enumerate(vocab_ids)}
    indptr, indices, src, n_empty_docs = [0], [], [], 0
    for s, code in zip(doc_sets, kept_src):
        cols = [col[t] for t in s if t in col]
        if not cols:
            n_empty_docs += 1
            continue
        indices.extend(cols)  # already ascending: s is sorted by id and col preserves the id order
        indptr.append(len(indices))
        src.append(code)
    indptr = np.asarray(indptr, dtype=np.int64)
    assert indptr[-1] < 2**31 - 1
    indptr = indptr.astype(np.int32)
    indices = np.asarray(indices, dtype=np.int32)
    vocab_df = np.array([df[int(t)] for t in vocab_ids], dtype=np.int32)
    n_docs = len(indptr) - 1
    shape = np.array([n_docs, len(vocab_ids)], dtype=np.int64)
    np.savez(OUT_NPZ, indptr=indptr, indices=indices, shape=shape, vocab_ids=vocab_ids, df=vocab_df,
             doc_source=np.asarray(src, dtype=np.int8))

    # ---- checks ----
    train_vocab = {}
    for n in ("clinc150", "banking77", "hwu64"):
        ids = tok.encode_batch(list(ds[n].texts["train"]), add_special_tokens=False)
        mine = len({t for e in ids for t in e.ids})
        study = json.loads(STUDY_META[n].read_text()).get("train_vocab") if STUDY_META[n].exists() else None
        train_vocab[n] = {"this_tokenizer": mine, "study_meta": study, "equal": mine == study}
    probes = {}
    for w in PROBES:
        ids = tok.encode(w, add_special_tokens=False).ids
        probes[w] = {"ids": ids, "single_token": len(ids) == 1,
                     "in_vocab": len(ids) == 1 and ids[0] in col,
                     "column": col.get(ids[0]) if len(ids) == 1 else None,
                     "df": int(df.get(ids[0], 0)) if len(ids) == 1 else None}
    q = np.quantile(vocab_df, [0.25, 0.5, 0.75]).tolist() if len(vocab_df) else []
    sizes = Counter(len(s) for s in doc_sets)
    manifest = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "inputs": {
            "dl_data_sources_sha256": {n: d.sources for n, d in ds.items()},
            "clinc_oos_plus": {"path": str(OOS_PLUS), "sha256": sha256_file(OOS_PLUS)},
            "tokenizer": {"hf_id": HF_ID, "revision": REV, "path": str(TOK_JSON), "sha256": sha256_file(TOK_JSON),
                          "add_special_tokens": False},
        },
        "key_normalization": "sha256(' '.join(re.findall(r\"[a-z0-9]+(?:'[a-z0-9]+)*\", NFKC(text) with "
                             "curly apostrophes -> ', lowercased)))",
        "blocklist_splits": block_info,
        "blocklist_distinct_keys": len(union_block),
        "corpus_sources": per_source,
        "dropped_per_eval_split": dropped_per_split,
        "dropped_total": int(sum(v["dropped"] for v in per_source.values())),
        "kept_total": len(kept_texts),
        "kept_distinct_keys": len(set(kept_keys)),
        "kept_duplicate_texts": len(kept_texts) - len(set(kept_keys)),
        "token_types_df_ge1": len(df),
        "vocab_size_df_ge2": int(len(vocab_ids)),
        "vocab_df_quartiles": q,
        "docs_in_matrix": int(n_docs),
        "docs_dropped_empty_after_vocab": int(n_empty_docs),
        "nnz": int(len(indices)),
        "mean_distinct_tokens_per_doc_before_vocab": round(float(np.mean([len(s) for s in doc_sets])), 3),
        "mean_vocab_tokens_per_doc": round(float(len(indices) / max(1, n_docs)), 3),
        "docs_with_1_token_before_vocab": int(sizes.get(1, 0)),
        "doc_source_codes": SOURCE_CODE,
        "check_train_vocab_vs_study_meta": train_vocab,
        "probes": probes,
        "outputs": {"npz": str(OUT_NPZ), "npz_sha256": sha256_file(OUT_NPZ)},
        "seconds": round(time.time() - t0, 2),
    }
    OUT_MAN.write_text(json.dumps(manifest, indent=1))
    short = {k: manifest[k] for k in ("created_utc", "kept_total", "dropped_total", "vocab_size_df_ge2",
                                      "token_types_df_ge1", "docs_in_matrix", "nnz", "seconds")}
    short["dropped_per_eval_split"] = {k: v["corpus_texts_matching"] for k, v in dropped_per_split.items()}
    short["train_vocab_equal_study_meta"] = {n: v["equal"] for n, v in train_vocab.items()}
    short["probes_in_vocab"] = sum(1 for v in probes.values() if v["in_vocab"])
    print(json.dumps(short))


if __name__ == "__main__":
    main()
