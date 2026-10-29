#!/usr/bin/env python
"""Test 5 harness (REGISTRATION_T5.md): one view x one dataset x one seed, VAL ONLY.

    python -I t5_tm.py --dataset clinc150|banking77|hwu64 --seed S [--smoke] [--codes-suffix SFX]
                       --view R1|R3a|potionV_K512|potionV_K86|ppmiV_K512|ppmiV_K86|wordsV

Data: dl.data.load. The test and oos_test splits are removed from the loaded object at once and any later access
fails an assert (the loader parses the files, nothing reads those splits). Tokens: the potion-base-32M tokenizer
(the file named in corpus_a.manifest.json, sha256 checked), add_special_tokens=False; check: the dataset's train
vocabulary has the study's size (5,291 / 2,326 / 4,373). A token has a code if it is one of the 4,564 corpus-(a)
vocabulary tokens and its row in the code table is not all zero; other tokens get nothing.
Views (code tables t5/codes_{table}{suffix}.npz, sha256 checked against t5/codes_manifest{suffix}.json):
- R1, wordsV: the sentence bits are the OR of its tokens' rows (wordsV: presence of each vocabulary token).
- R3a, potionV_K*, ppmiV_K*: the existsT encoder of exists_tm.py / exists2_tm.py: rows L2-normalized; K concepts
  (512 or 86) by dl.boolean._spherical_kmeans(Z, K, 25, gen) with a CUDA generator seeded 0, on the codes of the
  dataset's TRAINING tokens that have a code (sorted by token id, as exists_tm.py); per concept the max cosine over
  the sentence's coded tokens (a sentence without one gets 0.0, as exists_tm.pooled); dl.boolean.Thermo(8) fitted
  on the training sentences.
Classifier: dl.tm_init.make_tm(..., "half") with the TMConfig of exists2_tm.py (4,000 clauses, T 4000, s 8, 40
epochs, batch 128, hardest negative, negation, max_literals 0, budget exact / threshold, eval_every 2), trained by
tm.fit(Btr, ytr, Bva, yva); val accuracy at the best val epoch. GPU memory capped at 10% of the card.
One JSON line is appended to t5/results_t5.jsonl.
--smoke: 2,000 training examples (seeded permutation of the train split), 1 epoch, scored on the next 1,000
TRAIN examples (val is removed too), codes from the *_smoke tables unless --codes-suffix is given; nothing is
written. Never prints text.
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.sparse import csr_matrix  # noqa: E402

DL = Path(f"{_REPO_ROOT}/studies/decision_layer")
sys.path.insert(0, str(DL / "src"))
from dl.boolean import Thermo, _spherical_kmeans  # noqa: E402
from dl.data import load  # noqa: E402
from dl.tm import TMConfig  # noqa: E402
from dl.tm_init import make_tm  # noqa: E402

T5 = Path(__file__).resolve().parent
CORPUS_MAN = T5 / "corpus_a.manifest.json"
VIEWS = {"R1": ("R1", "bits", None), "R3a": ("R3a", "vec", 512), "potionV_K512": ("potionV", "vec", 512),
         "potionV_K86": ("potionV", "vec", 86), "ppmiV_K512": ("ppmiV", "vec", 512), "ppmiV_K86": ("ppmiV", "vec", 86),
         "wordsV": ("wordsV", "bits", None)}
FORBIDDEN = ("test", "oos_test")
STUDY_TRAIN_VOCAB = {"clinc150": 5291, "banking77": 2326, "hwu64": 4373}
MEM_FRACTION = 0.1


class Guard(dict):
    """A split dict that fails on any access to a removed split."""

    def __init__(self, d: dict, banned: set):
        super().__init__({k: v for k, v in d.items() if k not in banned})
        self.banned = set(banned)

    def __getitem__(self, k):
        assert k not in self.banned, f"split {k!r} must never be touched"
        return super().__getitem__(k)

    def get(self, k, default=None):
        assert k not in self.banned, f"split {k!r} must never be touched"
        return super().get(k, default)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def coded_rows(ids_list, row_of: dict, has: np.ndarray) -> list[list[int]]:
    out = []
    for ids in ids_list:
        rows = sorted({row_of[t] for t in ids if t in row_of})
        out.append([r for r in rows if has[r]])
    return out


def incidence(rows: list[list[int]], F: int) -> csr_matrix:
    indptr = np.zeros(len(rows) + 1, dtype=np.int64)
    indptr[1:] = np.cumsum([len(r) for r in rows])
    idx = np.fromiter((c for r in rows for c in r), dtype=np.int64, count=int(indptr[-1]))
    return csr_matrix((np.ones(len(idx), dtype=np.float32), idx, indptr), shape=(len(rows), F))


def or_bits(rows: list[list[int]], codes: np.ndarray, identity: bool) -> np.ndarray:
    S = incidence(rows, codes.shape[0])
    cf = None if identity else codes.astype(np.float32)
    out = np.zeros((len(rows), codes.shape[1]), dtype=np.uint8)
    for a in range(0, len(rows), 2000):
        blk = S[a : a + 2000]
        out[a : a + 2000] = (blk.toarray() if identity else np.asarray(blk @ cf)) > 0
    return out


def max_sims(rows: list[list[int]], S_tok: np.ndarray) -> np.ndarray:
    """Per sentence and concept, the max cosine over the sentence's coded tokens; 0.0 without one (exists_tm.pooled)."""
    out = np.zeros((len(rows), S_tok.shape[1]), dtype=np.float32)
    for a in range(0, len(rows), 2000):
        nz = [i for i in range(a, min(a + 2000, len(rows))) if rows[i]]
        if not nz:
            continue
        flat = np.fromiter((c for i in nz for c in rows[i]), dtype=np.int64)
        offs = np.cumsum([0] + [len(rows[i]) for i in nz[:-1]])
        out[nz] = np.maximum.reduceat(S_tok[flat], offs, axis=0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(STUDY_TRAIN_VOCAB))
    ap.add_argument("--view", required=True, choices=sorted(VIEWS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--codes-suffix", default=None, help="default: '' (real tables), '_smoke' with --smoke")
    args = ap.parse_args()
    sfx = args.codes_suffix if args.codes_suffix is not None else ("_smoke" if args.smoke else "")
    torch.cuda.set_per_process_memory_fraction(MEM_FRACTION)
    dev = "cuda"
    t0 = time.time()
    timings: dict = {}

    # ---- data: test / oos_test removed at once (and val too in a smoke run)
    ds = load(args.dataset)
    banned = set(FORBIDDEN) | ({"val", "oos_val"} if args.smoke else set())
    ds.texts, ds.labels = Guard(ds.texts, banned), Guard(ds.labels, banned)
    assert not (set(FORBIDDEN) & set(ds.texts.keys())) and not (set(FORBIDDEN) & set(ds.labels.keys()))
    n_classes = len(ds.intents)
    tr_texts_all, tr_y_all = list(ds.texts["train"]), np.asarray(ds.labels["train"])
    if args.smoke:
        perm = np.random.RandomState(0).permutation(len(tr_texts_all))
        itr, iva = perm[:2000], perm[2000:3000]
        tr_texts, ytr = [tr_texts_all[i] for i in itr], tr_y_all[itr]
        va_texts, yva = [tr_texts_all[i] for i in iva], tr_y_all[iva]
    else:
        tr_texts, ytr = tr_texts_all, tr_y_all
        va_texts, yva = list(ds.texts["val"]), np.asarray(ds.labels["val"])
    timings["load_s"] = round(time.time() - t0, 2)

    # ---- tokens
    t1 = time.time()
    from tokenizers import Tokenizer

    cman = json.loads(CORPUS_MAN.read_text())
    tok_path = Path(cman["inputs"]["tokenizer"]["path"])
    assert sha256_file(tok_path) == cman["inputs"]["tokenizer"]["sha256"], "tokenizer file changed"
    tok = Tokenizer.from_file(str(tok_path))
    enc = lambda texts: [e.ids for e in tok.encode_batch(texts, add_special_tokens=False)]  # noqa: E731
    full_train_vocab = len({t for ids in enc(tr_texts_all) for t in ids})
    assert full_train_vocab == STUDY_TRAIN_VOCAB[args.dataset], ("tokenizer differs from the study", full_train_vocab)
    ids_tr, ids_va = enc(tr_texts), enc(va_texts)
    timings["tokenize_s"] = round(time.time() - t1, 2)

    # ---- code table
    t1 = time.time()
    table, kind, K = VIEWS[args.view]
    cpath = T5 / f"codes_{table}{sfx}.npz"
    cmani = json.loads((T5 / f"codes_manifest{sfx}.json").read_text())
    csha = sha256_file(cpath)
    assert cmani["tables"][table]["sha256"] == csha, f"{cpath.name} differs from its manifest"
    zc = np.load(cpath)
    codes, vocab_ids = zc["codes"], zc["vocab_ids"].astype(np.int64)
    F = codes.shape[0]
    row_of = {int(t): i for i, t in enumerate(vocab_ids)}
    if kind == "vec":
        Zn = codes.astype(np.float32)
        nrm = np.linalg.norm(Zn, axis=1)
        has = nrm > 0
        Zn[has] /= nrm[has, None]
    else:
        has = codes.any(1)
    rows_tr, rows_va = coded_rows(ids_tr, row_of, has), coded_rows(ids_va, row_of, has)
    n_val_tok = sum(len(x) for x in ids_va)
    n_val_tok_coded = sum(1 for x in ids_va for t in x if t in row_of and has[row_of[t]])
    timings["codes_s"] = round(time.time() - t1, 2)

    # ---- sentence bits
    t1 = time.time()
    extra: dict = {}
    if kind == "bits":
        Btr = or_bits(rows_tr, codes, identity=(table == "wordsV"))
        Bva = or_bits(rows_va, codes, identity=(table == "wordsV"))
    else:
        train_rows = sorted({r for rr in rows_tr for r in rr})
        Z = torch.as_tensor(Zn[train_rows], device=dev)
        gen = torch.Generator(device=dev)
        gen.manual_seed(0)
        tk = time.time()
        Cn = _spherical_kmeans(Z, K, 25, gen)
        timings["kmeans_s"] = round(time.time() - tk, 2)
        S_tok = (torch.as_tensor(Zn, device=dev) @ Cn.T).cpu().numpy()
        sims_tr, sims_va = max_sims(rows_tr, S_tok), max_sims(rows_va, S_tok)
        th = Thermo(8).fit(sims_tr)
        Btr, Bva = th.transform(sims_tr), th.transform(sims_va)
        extra = {"K": int(Cn.shape[0]), "kmeans_tokens": len(train_rows)}
    timings["encode_s"] = round(time.time() - t1, 2)

    # ---- classifier
    cfg = TMConfig(n_clauses=4000, T=4000, s=8.0, epochs=1 if args.smoke else 40, batch_size=128,
                   neg_sampling="hardest", negation=True, max_literals=0, budget_mode="exact",
                   budget_select="threshold", max_negated=0, clause_drop=0.0, half=True, seed=args.seed, eval_every=2)
    tm = make_tm(n_classes, Btr.shape[1], cfg, "half")
    tm.fit(Btr, ytr, Bva, yva)
    timings["train_s"] = round(tm.train_seconds, 1)
    timings["total_s"] = round(time.time() - t0, 1)
    assert not (set(FORBIDDEN) & set(ds.texts.keys())), "a forbidden split came back"
    acc_key = "heldout_train_acc" if args.smoke else "val_acc"
    rec = {
        "registration": "REGISTRATION_T5.md", "smoke": args.smoke, "dataset": args.dataset, "view": args.view,
        "table": table, "codes_file": cpath.name, "codes_sha256": csha[:16], "seed": args.seed,
        "n_bits": int(Btr.shape[1]), "on_share_train": round(float(Btr.mean()), 4),
        "on_share_val": round(float(Bva.mean()), 4), "bits_on_per_val_sentence": round(float(Bva.sum(1).mean()), 2),
        "val_tokens": int(n_val_tok), "val_tokens_without_code_share": round(1 - n_val_tok_coded / max(1, n_val_tok), 4),
        "val_sentences_without_code": int(sum(1 for r in rows_va if not r)),
        "vocab_rows_with_code": int(has.sum()), **extra,
        acc_key: round(float(tm.best_val), 4), "best_epoch": getattr(tm, "best_epoch", None),
        "n_train": int(len(ytr)), "n_scored": int(len(yva)), "timings": timings,
        "curve": [(h["epoch"], round(h["val_acc"], 4)) for h in tm.history], "stats": tm.stats(),
        "cfg": asdict(cfg), "tokenizer_train_vocab_check": full_train_vocab,
        "sha256": {"t5_tm.py": sha256_file(Path(__file__).resolve())[:16],
                   **{f"dl/{m}": sha256_file(DL / "src" / "dl" / m)[:16] for m in ("tm.py", "tm_init.py", "boolean.py", "data.py")}},
        "versions": {"torch": torch.__version__, "numpy": np.__version__}, "gpu": torch.cuda.get_device_name(0),
    }
    if not args.smoke:
        with open(T5 / "results_t5.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k not in ("cfg", "curve", "stats")}))


if __name__ == "__main__":
    main()
