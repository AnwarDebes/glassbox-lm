#!/usr/bin/env python
"""Fresh intent test set, step 0 (CPU; before any registration names MTOP): the overlap of MTOP English's texts with
the texts this project has trained, tuned or evaluated on. It reads the MTOP utterance column only, never the intent,
slot, domain or decoupled-form columns, and scores nothing.

    python -I -B fresh/mtop_overlap.py --zip data_untrusted/mtop/s3v_qLlhqQW2ypnQQXRE/mtop.zip --out fresh/mtop_overlap.json
    python -I -B fresh/mtop_overlap.py --selftest

Keys of a text (build_corpus_c4.norm: NFKC, curly quotes straightened, lowercased, its words joined by one space):
- exact: the norm itself;
- order-free: the sorted words of the norm (the same words in any order);
- one-edit: the word sequences within one word insertion, deletion or substitution (word-level Levenshtein <= 1),
  found by deletion keys (a sequence and every copy with one word removed). Texts of fewer than 3 words are left out of
  the one-edit key (too many chance matches).
Pools, each counted on its own:
- ours_train_val: CLINC150 (train, val, oos_train, oos_val), Banking77 and HWU64 (train, val), MASSIVE en-US (train,
  val), via dl.data.load;
- ours_test_texts: the test splits of the same sets, text only (no label read);
- paraphrases: 3c's generated rows (tu/c3/gen.jsonl, every generated paraphrase), kept_F, kept_C and the EDA twin
  (tu/c3x/twin_kept.jsonl);
- c4_fit: 3a's C4 fit and held-out sentences (tu/a3/c4_sent_{train,held}_f.txt), streamed;
- other_eval: every other text of build_corpus_c4.eval_texts() (CLINC OOS+, MASSIVE nb, TREC, SST-2, MR, R52, R8, ABCD,
  STAR);
- mtop_internal: MTOP's own train and eval texts, for its test texts (descriptive).
The output gives counts and MTOP indices per split, key and pool, and examples for MTOP train and eval only (no MTOP
test text is printed). Refuses to overwrite.
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))

import argparse
import hashlib
import json
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent  # glassbox-lm/studies/text_understanding/fresh
EMB = HERE.parent
TU = EMB / "tu"
SPLITS = ("train", "eval", "test")
UTT = 3  # the utterance column of MTOP's 8-column TSV (0-based); no other column is read


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def mtop_texts(zip_path: Path) -> dict[str, list[str]]:
    z = zipfile.ZipFile(zip_path)
    out = {}
    for sp in SPLITS:
        lines = z.read(f"mtop/en/{sp}.txt").decode("utf-8").split("\n")
        rows = [ln.split("\t") for ln in lines if ln.strip()]
        assert all(len(r) == 8 for r in rows), f"{sp}: a row without 8 columns"
        out[sp] = [r[UTT] for r in rows]
    return out


def dels(words: tuple) -> list[tuple]:
    return [words[:i] + words[i + 1 :] for i in range(len(words))]


def key_sets(texts, norm) -> dict[str, set]:
    ex, of, oe = set(), set(), set()
    for t in texts:
        n = norm(t)
        if not n:
            continue
        w = tuple(n.split())
        ex.add(n)
        of.add(tuple(sorted(w)))
        if len(w) >= 3:
            oe.add(w)
            oe.update(dels(w))
    return {"exact": ex, "order_free": of, "one_edit": oe}


def hits(texts, ks: dict[str, set], norm) -> dict[str, list[int]]:
    out = {"exact": [], "order_free": [], "one_edit": []}
    for i, t in enumerate(texts):
        n = norm(t)
        if not n:
            continue
        w = tuple(n.split())
        if n in ks["exact"]:
            out["exact"].append(i)
        if tuple(sorted(w)) in ks["order_free"]:
            out["order_free"].append(i)
        if len(w) >= 3 and (w in ks["one_edit"] or any(d in ks["one_edit"] for d in dels(w))):
            out["one_edit"].append(i)
    return out


def stream_hits(paths, mt: dict[str, list[str]], norm) -> tuple[dict, int]:
    """One pass over large sentence files; only MTOP's keys are held in memory."""
    maps = {}
    for sp, texts in mt.items():
        em, om, dm = {}, {}, {}
        for i, t in enumerate(texts):
            n = norm(t)
            if not n:
                continue
            w = tuple(n.split())
            em.setdefault(n, []).append(i)
            om.setdefault(tuple(sorted(w)), []).append(i)
            if len(w) >= 3:
                for k in [w, *dels(w)]:
                    dm.setdefault(k, []).append(i)
        maps[sp] = (em, om, dm)
    out = {sp: {"exact": set(), "order_free": set(), "one_edit": set()} for sp in mt}
    n_lines = 0
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                n_lines += 1
                n = norm(line.rstrip("\n"))
                if not n:
                    continue
                w = tuple(n.split())
                o = tuple(sorted(w))
                cand = [w, *dels(w)] if 3 <= len(w) <= 40 else []
                for sp, (em, om, dm) in maps.items():
                    if n in em:
                        out[sp]["exact"].update(em[n])
                    if o in om:
                        out[sp]["order_free"].update(om[o])
                    for k in cand:
                        if k in dm:
                            out[sp]["one_edit"].update(dm[k])
    return {sp: {k: sorted(v) for k, v in d.items()} for sp, d in out.items()}, n_lines


def selftest() -> None:
    import re

    W = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*")
    norm = lambda t: " ".join(W.findall(t.lower()))
    pool = ["Set an alarm for 7 am", "what is the weather today", "play some jazz music please"]
    ks = key_sets(pool, norm)
    q = ["set an ALARM for 7 am", "today the weather is what", "play some jazz music", "play some rock music please",
         "call mom now", "set alarm"]
    h = hits(q, ks, norm)
    assert h["exact"] == [0], h
    assert h["order_free"] == [0, 1], h
    assert h["one_edit"] == [0, 2, 3], h  # a deletion and a substitution are one edit; a 2-word text never matches
    tmp = Path(sys.argv[0]).resolve().parent / ".selftest_pool.txt"
    assert not tmp.exists()
    try:
        tmp.write_text("\n".join(pool) + "\n")
        sh, n = stream_hits([tmp], {"x": q}, norm)
        assert n == 3 and sh["x"] == {"exact": [0], "order_free": [0, 1], "one_edit": [0, 2, 3]}, sh
    finally:
        tmp.unlink()
    assert hits(["call mom now"], ks, norm) == {"exact": [], "order_free": [], "one_edit": []}, "negative check"
    print("SELFTEST PASSED")


def main() -> None:
    if sys.argv[1:] == ["--selftest"]:
        selftest()
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists(), f"{args.out} exists"
    t0 = time.time()
    sys.path.insert(0, str(EMB / "w1"))
    sys.path.insert(0, f"{_REPO_ROOT}/studies/decision_layer/src")
    import build_corpus_c4 as B
    from dl.data import load

    mt = mtop_texts(args.zip)
    files = {"mtop.zip": sha256_file(args.zip)}
    sets = {n: load(n) for n in ("clinc150", "banking77", "hwu64", "massive_en")}
    tv, te = [], []
    for n, d in sets.items():
        files[f"dl.data:{n}"] = json.dumps(d.sources, sort_keys=True)
        for sp, texts in d.texts.items():
            (te if "test" in sp else tv).extend(texts)
    para = []
    for rel, field in (("c3/gen.jsonl", "paraphrases"), ("c3/kept_F.jsonl", "text"), ("c3/kept_C.jsonl", "text"),
                       ("c3x/twin_kept.jsonl", "text")):
        p = TU / rel
        for x in p.read_text(encoding="utf-8").splitlines():
            if x.strip():
                v = json.loads(x)[field]  # gen.jsonl: every generated paraphrase; the kept files: the kept text
                para += v if isinstance(v, list) else [v]
        files[f"tu/{rel}"] = sha256_file(p)
    ev, evf = B.eval_texts()
    files.update({f"eval_texts:{k}": v for k, v in evf.items()})
    ours = ("clinc150/", "banking77/", "hwu64/", "massive_en/")
    other = [t for k, v in ev.items() if not k.startswith(ours) for t in v]
    pools = {"ours_train_val": tv, "ours_test_texts": te, "paraphrases": para, "other_eval": other}
    res = {"mtop_sizes": {sp: len(v) for sp, v in mt.items()}, "pool_sizes": {k: len(v) for k, v in pools.items()},
           "hits": {sp: {} for sp in SPLITS}}
    for name, texts in pools.items():
        ks = key_sets(texts, B.norm)
        for sp in SPLITS:
            res["hits"][sp][name] = hits(mt[sp], ks, B.norm)
    ks = key_sets(mt["train"] + mt["eval"], B.norm)
    res["hits"]["test"]["mtop_internal"] = hits(mt["test"], ks, B.norm)
    c4 = [TU / "a3" / "c4_sent_train_f.txt", TU / "a3" / "c4_sent_held_f.txt"]
    sh, n_c4 = stream_hits(c4, mt, B.norm)
    for sp in SPLITS:
        res["hits"][sp]["c4_fit"] = sh[sp]
    files.update({f"tu/a3/{p.name}": sha256_file(p) for p in c4})
    res["pool_sizes"]["c4_fit"] = n_c4
    res["counts"] = {sp: {pool: {k: len(v) for k, v in h.items()} for pool, h in d.items()} for sp, d in res["hits"].items()}
    res["examples"] = {sp: {pool: {k: [mt[sp][i] for i in v[:5]] for k, v in h.items()} for pool, h in res["hits"][sp].items()}
                       for sp in ("train", "eval")}
    res.update({"files": files, "mtop_overlap.py": sha256_file(Path(__file__).resolve()),
                "seconds": round(time.time() - t0, 1), "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
    args.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({"mtop_sizes": res["mtop_sizes"], "pool_sizes": res["pool_sizes"], "counts": res["counts"],
                      "seconds": res["seconds"]}, indent=1))


if __name__ == "__main__":
    main()
