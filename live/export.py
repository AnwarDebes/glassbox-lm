#!/usr/bin/env python3
"""Build the data of the glass box page (live/) with the study's own code.

1. Ten public-domain books are downloaded from Project Gutenberg (or read from --cache), their
   headers, licence text, contents pages and illustration captions are removed, and every book is
   cut into chapters. A chapter is one document.
2. Tokens: lowercase words (inner apostrophes kept), numbers, and the marks . , ; : ! ? Quotation
   marks, dashes, brackets and underscores are dropped. The vocabulary is every word of the
   training chapters, most frequent first, followed by 64 document-word slots that never occur in
   training: a word that the vocabulary does not hold takes the next free slot of its document, so
   a repeated new word keeps one id (the page does the same with typed text).
3. Splits by the residue of docs/11 section 11.6, mix64(i XOR 0xC0FFEE) mod 100 over the global
   chapter index i: residues 90 to 99 are held-out chapters, except the first chapter of each book,
   which always stays in training. Among the training chapters, residues 0 to 9 (of the training
   index) are the fit documents H of the EM (the study uses residue 0 of 28,472 articles; a corpus
   of a few hundred chapters needs a larger share).
4. `tmlm.cvtm.model.CVTMCore.fit` (numpy backend, components S, D0..D4, P, whole-document window,
   no pruning) fits the CVTM-core: KN-5 without H, MAP-EM weights on H, KN-5 on all training
   chapters. The held-out chapters are scored with `CVTMCore.gold`, the KN orders 1 to 5 with
   `KNModel.logprob`, and the vote form is checked against `KNModel.prob` (H8.2 of the study).

Writes corpus.json, tokens.u16.gz, core.json and study.json (the 91 registered ids of
tm_language_model with their statuses, read from its results page) next to this file, and
tests/golden.json, the reference values that tests/model.test.mjs checks the browser model against.

    python live/export.py [--cache DIR]
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
import tempfile
import time
import unicodedata
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "studies" / "tm_language_model" / "src"))

from tmlm.cvtm import mixture, model as cvtm_model, protocol  # noqa: E402
from tmlm.engine.rng import mix64  # noqa: E402

N_SLOTS = 64
VAL_RESIDUES = range(90, 100)
H_RESIDUES = tuple(range(10))
SALT = 0xC0FFEE

# Gutenberg id, title, author, year, heading rule, title lines after a heading, first body line
BOOKS = [
    (98, "A Tale of Two Cities", "Charles Dickens", 1859, r"^CHAPTER [IVXL]+\.$", 1, None),
    (1342, "Pride and Prejudice", "Jane Austen", 1813, r"^CHAPTER [IVXLC]+\.$", 0, r"^It is a truth universally acknowledged"),
    (11, "Alice's Adventures in Wonderland", "Lewis Carroll", 1865, r"^CHAPTER [IVXL]+\.$", 1, None),
    (1661, "The Adventures of Sherlock Holmes", "Arthur Conan Doyle", 1892, r"^[IVX]+\. [A-Z][A-Z’' \-]+$", 0, None),
    (84, "Frankenstein", "Mary Shelley", 1818, r"^(Letter \d+|Chapter \d+)$", 0, None),
    (43, "The Strange Case of Dr Jekyll and Mr Hyde", "Robert Louis Stevenson", 1886,
     r"^(STORY OF THE DOOR|SEARCH FOR MR\. HYDE|DR\. JEKYLL WAS QUITE AT EASE|THE CAREW MURDER CASE|"
     r"INCIDENT OF THE LETTER|INCIDENT OF DR\. LANYON|INCIDENT AT THE WINDOW|THE LAST NIGHT|"
     r"DR\. LANYON’S NARRATIVE|HENRY JEKYLL’S FULL STATEMENT OF THE CASE)$", 0, None),
    (35, "The Time Machine", "H. G. Wells", 1895, r"^ [IVX]+\.$", 1, None),
    (46, "A Christmas Carol", "Charles Dickens", 1843, r"^STAVE [IVX]+:", 0, None),
    (16, "Peter Pan", "J. M. Barrie", 1911, r"^Chapter [IVXL]+\.$", 1, None),
    (55, "The Wonderful Wizard of Oz", "L. Frank Baum", 1900, r"^Chapter [IVXL]+$", 1, None),
]

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*|[.,;:!?]")


# ------------------------------------------------------------------ text
def fetch(gid: int, cache: Path) -> str:
    path = cache / f"pg{gid}.txt"
    if not path.exists():
        url = f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"
        with urllib.request.urlopen(url, timeout=60) as r:
            path.write_bytes(r.read())
    return path.read_text(encoding="utf-8").replace("\r", "")


def body_lines(text: str) -> list[str]:
    a = text.find("*** START OF")
    b = text.find("*** END OF")
    return text[text.find("\n", a) + 1:b].split("\n")


def drop_illustrations(lines: list[str]) -> list[str]:
    out, inside = [], False
    for line in lines:
        s = line.strip()
        if not inside and s.startswith("[Illustration"):
            inside = not s.endswith("]")
            continue
        if inside:
            if s.endswith("]"):
                inside = False
            continue
        out.append(line)
    return out


def chapters(gid: int, heading: str, title_lines: int, first_line: str | None, lines: list[str]) -> list[tuple[str, list[str]]]:
    """(label, lines) of every chapter of one book."""
    lines = drop_illustrations(lines)
    head = re.compile(heading)
    docs: list[tuple[str, list[str]]] = []
    cur: list[str] | None = None
    raw, title, part = "", "", 0
    skip = 0
    if first_line is not None:                        # Pride and Prejudice: chapter I has no heading line
        start = next(i for i, l in enumerate(lines) if re.match(first_line, l))
        cur, raw = [], "Chapter I"
        lines = lines[start:]

    def close() -> None:
        if cur is not None and any(x.strip() for x in cur):
            docs.append((chapter_label(gid, raw, title, part), cur))

    for line in lines:
        s = line.rstrip()
        if gid == 98 and re.match(r"^Book the (First|Second|Third)", s.strip()):
            part += 1 if cur is not None or part == 0 else 0
            continue
        if head.match(s):
            close()
            cur, raw, title, skip = [], s.strip(), "", title_lines
            continue
        if cur is None:
            continue
        if skip and s.strip():
            title = s.strip()
            skip -= 1
            continue
        if gid == 1661 and re.match(r"^[IVX]+\.$", s.strip()):    # numbered sections inside a story
            continue
        cur.append(s)
    close()
    return docs


SMALL = {"a", "an", "and", "at", "by", "for", "from", "in", "of", "on", "or", "the", "to", "with"}


def title_case(text: str) -> str:
    """Title case for headings set in capitals; mixed-case text is returned as it is."""
    text = text.replace("’", "'").strip()
    letters = [c for c in text if c.isalpha()]
    if not letters or sum(c.isupper() for c in letters) < 0.8 * len(letters):
        return text
    words = text.lower().split()
    out = []
    for i, w in enumerate(words):
        if re.fullmatch(r"[ivxlc]+\.?", w) and i == 0:
            out.append(w.upper())
        elif i > 0 and w in SMALL:
            out.append(w)
        else:
            out.append("-".join(x[:1].upper() + x[1:] for x in w.split("-")))
    return " ".join(out)


def chapter_label(gid: int, raw: str, title: str, part: int) -> str:
    """A readable label: "Chapter IV: The Preparation", "Stave I: Marley's Ghost", a story title."""
    raw = raw.strip()
    title = title_case(title) if title else ""
    m = re.match(r"^(CHAPTER|Chapter|STAVE|Letter)\s+([IVXLC]+|\d+)[.:]?\s*(.*)$", raw)
    if m:
        kind = m.group(1).capitalize()
        rest = title_case(m.group(3)) if m.group(3) else title
        label = f"{kind} {m.group(2)}" + (f": {rest}" if rest else "")
        return (f"Book {part}, " if gid == 98 and part else "") + label
    m = re.match(r"^([IVX]+)\.\s*(.*)$", raw)
    if m:                                             # The Time Machine and the Sherlock Holmes stories
        if gid == 1661:
            return title_case(m.group(2))
        return f"Chapter {m.group(1)}" + (f": {title}" if title else "")
    return title_case(raw)


def normalise(text: str) -> str:
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = text.replace("_", " ").replace("--", " ").replace("\u2014", " ").replace("\u2013", " ")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("æ", "ae").replace("Æ", "Ae").replace("œ", "oe").replace("Œ", "Oe")
    return text.lower()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(normalise(text))


def study_ids() -> list[dict]:
    """Every registered id of tm_language_model with its status, from the study's results page."""
    out, seen = [], set()
    page = ROOT / "studies" / "tm_language_model" / "docs" / "results.md"
    for line in page.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\| (H\d+\.\d+) \| (.*?) \| (met|not met|not judged|not run)(?: \(.*?\))? \| (.*?) \|", line)
        if m and m.group(1) not in seen:
            seen.add(m.group(1))
            out.append({"id": m.group(1), "text": m.group(2), "status": m.group(3), "qualifier": m.group(4)})
    out.sort(key=lambda r: tuple(int(x) for x in r["id"][1:].split(".")))
    return out


def residue(i: int) -> int:
    return mix64(i ^ SALT) % 100


# ------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", type=Path, default=None, help="folder for the downloaded books")
    args = ap.parse_args(argv)
    t_start = time.perf_counter()
    cache = args.cache or Path(tempfile.mkdtemp(prefix="gutenberg-"))
    cache.mkdir(parents=True, exist_ok=True)

    books, docs = [], []
    for b, (gid, title, author, year, heading, tl, first) in enumerate(BOOKS):
        text = fetch(gid, cache)
        chs = chapters(gid, heading, tl, first, body_lines(text))
        books.append({"gutenberg": gid, "title": title, "author": author, "year": year, "chapters": len(chs)})
        for c, (label, lines) in enumerate(chs):
            toks = tokenize("\n".join(lines))
            docs.append({"book": b, "chapter": c, "label": label, "tokens": toks})
        print(f"{title}: {len(chs)} chapters, {sum(len(d['tokens']) for d in docs if d['book'] == b):,} tokens")

    for i, d in enumerate(docs):
        d["residue"] = residue(i)
        d["split"] = "heldout" if (d["residue"] in VAL_RESIDUES and d["chapter"] > 0) else "train"
    train = [d for d in docs if d["split"] == "train"]
    held = [d for d in docs if d["split"] == "heldout"]

    counts: dict[str, int] = {}
    for d in train:
        for t in d["tokens"]:
            counts[t] = counts.get(t, 0) + 1
    words = sorted(counts, key=lambda w: (-counts[w], w))
    vocab = words + [f"<w{j}>" for j in range(N_SLOTS)]
    V = len(vocab)
    if V > 32768:
        raise SystemExit(f"vocabulary {V} exceeds the 15-bit token ids of KNModel")
    index = {w: i for i, w in enumerate(words)}
    slot0 = len(words)

    def encode(toks: list[str]) -> tuple[list[int], list[str]]:
        ids, new = [], []
        for t in toks:
            if t in index:
                ids.append(index[t])
                continue
            if t not in new:
                new.append(t)
            ids.append(slot0 + min(new.index(t), N_SLOTS - 1))
        return ids, new

    stream, ds_all = [], [0]
    for d in docs:
        ids, new = encode(d["tokens"])
        d["ids"] = ids
        d["new_words"] = new
        stream.extend(ids)
        ds_all.append(len(stream))
    tok_all = np.asarray(stream, dtype=np.uint16)

    def concat(sel):
        t, s = [], [0]
        for d in sel:
            t.extend(d["ids"])
            s.append(len(t))
        return np.asarray(t, dtype=np.int64), np.asarray(s, dtype=np.int64)

    tr_tok, tr_ds = concat(train)
    ho_tok, ho_ds = concat(held)
    print(f"vocabulary {V} ({len(words)} words + {N_SLOTS} slots); training {len(train)} chapters, {tr_tok.shape[0]:,} tokens; "
          f"held-out {len(held)} chapters, {ho_tok.shape[0]:,} tokens")

    # ---- the CVTM-core, fitted by the study's code
    protocol.H_RESIDUES = H_RESIDUES
    t0 = time.perf_counter()
    core = cvtm_model.CVTMCore.fit(tr_tok, tr_ds, vocab=V, window=None, components="full", backend="numpy",
                                   expect_h=False, log=print)
    fit_s = time.perf_counter() - t0
    kn = core.kn
    rec = core.gold(ho_tok, ho_ds)
    lnp_s = np.asarray(rec.lnp_s, dtype=np.float64)

    # vote form against the recursion (H8.2 of the study), on every held-out position
    ctx, valid = kn.contexts(ho_tok, ho_ds, np.arange(ho_tok.shape[0]), None, 5)
    p_rec = kn.prob(ctx, ho_tok, valid, 5)
    h82 = float(np.max(np.abs(lnp_s - np.log(p_rec))))
    order_bits = {}
    for k in range(1, 6):
        order_bits[k] = float(-kn.logprob(ho_tok, ho_ds, k).sum())
    core_bits = float(-np.log2(rec.p_gold).sum())
    n_ho = int(ho_tok.shape[0])
    print(f"vote form vs recursion: max |d ln P| = {h82:.3e} over {n_ho} positions")
    for k in range(1, 6):
        print(f"  KN order {k}: {order_bits[k] / n_ho:.4f} bits per token, perplexity {2 ** (order_bits[k] / n_ho):.2f}")
    print(f"  CVTM-core: {core_bits / n_ho:.4f} bits per token, perplexity {2 ** (core_bits / n_ho):.2f}")

    # ---- reference values for the browser test
    rng = np.random.default_rng(20261021)
    sample = np.sort(rng.choice(n_ho, size=min(2500, n_ho), replace=False))
    top_pos = np.sort(rng.choice(n_ho, size=40, replace=False))
    top_ids, top_p = core.topk(ho_tok, ho_ds, top_pos, k=8, rec=rec)
    feats = rec.feats
    golden = {
        "note": "reference values from the study's code (tmlm.cvtm) on the held-out chapters; positions index the held-out stream",
        "n_positions": n_ho,
        "h82_max_abs_diff": h82,
        "order_bits": {str(k): v for k, v in order_bits.items()},
        "core_bits": core_bits,
        "positions": sample.tolist(),
        "ln_p_kn": [float(x) for x in lnp_s[sample]],
        "p_core": [float(x) for x in rec.p_gold[sample]],
        "k_star": [int(x) for x in rec.kstar[sample]],
        "bucket": [int(x) for x in rec.bucket[sample]],
        "n_m": [[int(feats.count(m)[p]) for m in range(5)] for p in sample],
        "ptr_len": [int(feats.ptr_len[p]) for p in sample],
        "ptr_cand": [int(feats.ptr_cand[p]) for p in sample],
        "top_positions": top_pos.tolist(),
        "top_ids": top_ids.tolist(),
        "top_p": [[float(x) for x in row] for row in top_p],
        "discounts": {str(k): {"raw": list(kn.raw[k].d), "cont": (list(kn.cont[k].d) if kn.cont[k] is not None else None)}
                      for k in range(1, 6)},
        "n_types": [int(x) for x in kn.n_types],
    }

    # ---- files
    doc_rows = []
    for i, d in enumerate(docs):
        doc_rows.append({"book": d["book"], "label": d["label"], "start": ds_all[i], "length": ds_all[i + 1] - ds_all[i],
                         "split": d["split"], "residue": d["residue"], "new_words": d["new_words"]})
    corpus = {
        "format": "glassbox-corpus-1",
        "tokenizer": {"regex": TOKEN_RE.pattern, "lowercase": True,
                      "dropped": "quotation marks, dashes, brackets, underscores; accents folded"},
        "books": books,
        "docs": doc_rows,
        "vocab": vocab,
        "n_words": len(words),
        "n_slots": N_SLOTS,
        "n_tokens": int(tok_all.shape[0]),
        "tokens_sha256": hashlib.sha256(tok_all.astype("<u2").tobytes()).hexdigest(),
    }
    (HERE / "corpus.json").write_text(json.dumps(corpus, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    with gzip.GzipFile(HERE / "tokens.u16.gz", "wb", compresslevel=9, mtime=0) as f:
        f.write(tok_all.astype("<u2").tobytes())
    trace = core.fit_info.get("em_trace", {})
    core_doc = {
        "format": "glassbox-core-1",
        "estimator": "CVTM-core of docs/11 section 11 (suffix clauses = interpolated modified Kneser-Ney, order 5; "
                     "document clauses D0..D4; pointer 5..64; bucketed MAP-EM mixture), fitted by tmlm.cvtm.model.CVTMCore",
        "components": list(mixture.COMPONENTS),
        "window": None,
        "ptr_min": mixture.PTR_MIN,
        "ptr_max": cvtm_model.PTR_MAX,
        "em_iters": cvtm_model.EM_ITERS,
        "prior": cvtm_model.PRIOR,
        "fit_residues": list(H_RESIDUES),
        "heldout_residues": [VAL_RESIDUES.start, VAL_RESIDUES.stop - 1],
        "weights": [[float(x) for x in row] for row in core.weights],
        "pooled_weights": trace.get("pooled_weights"),
        "fit": {"train_chapters": len(train), "train_tokens": int(tr_tok.shape[0]), "H_chapters": int(core.fit_info["H_docs"]),
                "H_tokens": int(core.fit_info["H_tokens"]), "heldout_chapters": len(held), "heldout_tokens": n_ho,
                "fit_seconds": round(fit_s, 1)},
        "heldout": {"tokens": n_ho, "order_bits": {str(k): v for k, v in order_bits.items()}, "core_bits": core_bits,
                    "vote_vs_recursion_max_abs_diff": h82},
        "state_hash": core.state_hash(),
    }
    (HERE / "core.json").write_text(json.dumps(core_doc, separators=(",", ":")), encoding="utf-8")
    ids = study_ids()
    if len(ids) != 91:
        raise SystemExit(f"expected 91 registered ids on the results page, found {len(ids)}")
    (HERE / "study.json").write_text(json.dumps({"source": "studies/tm_language_model/docs/results.md", "ids": ids},
                                                separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    (HERE / "tests" / "golden.json").write_text(json.dumps(golden, separators=(",", ":")), encoding="utf-8")
    print(f"wrote corpus.json, tokens.u16.gz, core.json, tests/golden.json in {time.perf_counter() - t_start:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
