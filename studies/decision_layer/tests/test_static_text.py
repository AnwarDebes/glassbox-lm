"""The no-network text bits (src/dl/static_text.py) and their place in ABCDView (src/dl/abcd.py).

    HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 .venv-dl/bin/python tests/test_static_text.py
    DL_TEST_GPU=1 ... (under scripts/gpu_run.sh shared): also the default bge-large view, pristine vs new abcd.py

CPU part:
- the table: pinned files (checked at load) and the table hash of both caches;
- mean equals SentenceTransformer.encode(normalize_embeddings=True) within 1e-6 on every unique dev text;
- the caches' rows align with the records (cache vector = mean of the record's text);
- transform (lookup + OR) equals transform_max (max cosine, then thresholds) exactly on every dev text,
  for both fields and both datasets; two fits give identical concepts, thresholds and codes;
- ABCDView's mem and words blocks equal the cached readable bits; the static parts come last in the fixed
  order; an unknown part raises.
"""

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dl.abcd import PARTS, ABCDView, load_examples  # noqa: E402
from dl.static_text import EMPTY, ExistsT, table  # noqa: E402


def log(*a):
    print(*a, flush=True)


def test_table_and_caches():
    tb = table()
    for ds in ("abcd", "star"):
        z = np.load(ROOT / "data" / "emb" / f"{ds}__potion-32m.npz")
        prov = json.loads(str(z["provenance"]))
        assert prov["table_sha256"] == tb.table_sha256, ds
        dv, _ = load_examples("dev", ds)
        for fld in ("cust", "recent"):
            texts = [r[fld] or EMPTY for r in dv]
            uniq = sorted(set(texts))
            M = tb.mean(uniq)
            ref = tb.st.encode(uniq, batch_size=256, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
            gap = float(np.abs(M - ref).max())
            assert gap <= 1e-6, (ds, fld, gap)
            pos = {t: i for i, t in enumerate(uniq)}
            assert np.array_equal(z[f"dev__{fld}"], M[[pos[t] for t in texts]].astype(np.float32)), (ds, fld, "rows")
            log(f"table/cache {ds} {fld}: {len(uniq)} unique dev texts, gap {gap:.2e}, rows aligned")


def test_exists_or_equals_max():
    for ds in ("abcd", "star"):
        tr, _ = load_examples("train", ds)
        dv, _ = load_examples("dev", ds)
        for fld in ("cust", "recent"):
            x = ExistsT(what=fld).fit([r[fld] for r in tr])
            a, b = x.transform([r[fld] for r in dv]), x.transform_max([r[fld] for r in dv])
            assert a.shape == b.shape == (len(dv), x.n_bits) and np.array_equal(a, b), (ds, fld)
            on = a.mean()
            log(f"existsT {ds} {fld}: {x.n_bits} bits, OR == max on all {len(dv)} dev texts, {on:.3f} of bits on, "
                f"concepts {x.info['concepts_sha256'][:16]}")
            if ds == "star" and fld == "recent":
                y = ExistsT(what=fld).fit([r[fld] for r in tr])
                assert np.array_equal(x.C, y.C) and all(np.array_equal(p, q) for p, q in zip(x.th, y.th))
                assert np.array_equal(x.packed, y.packed) and x.info == y.info
                log("existsT: two fits identical (concepts, thresholds, codes)")


def test_view_blocks():
    assert PARTS[-2:] == ("xcust", "xrecent")
    try:
        ABCDView(parts=("mem", "wrods"))
        raise AssertionError("an unknown part must raise")
    except ValueError:
        pass
    ds = "star"
    tr, blist = load_examples("train", ds)
    dv, _ = load_examples("dev", ds)
    cache = np.load(ROOT / "data" / ds / "bits_mem-intent-step-words_dev.npz")
    v = ABCDView(parts=("mem", "words"), seed=0).fit(tr, {}, blist)
    X = v.transform(dv, {})
    n_mem = 2 * len(blist) + 6
    assert np.array_equal(X[:, :n_mem], cache["X"][:, :n_mem])
    assert np.array_equal(X[:, n_mem:], cache["X"][:, -1000:])
    log("view: mem and words blocks equal the cached readable bits (STAR dev); unknown part raises")
    w = ABCDView(parts=("xrecent", "mem", "words"), seed=0).fit(tr, {}, blist)
    Xw = w.transform(dv, {})
    assert np.array_equal(Xw[:, : X.shape[1]], X), "static parts must come after mem and words"
    assert w.names[: len(v.names)] == v.names and w.names[len(v.names)].startswith("some word of the last 3 turns")
    log(f"view: xrecent appended after mem and words ({Xw.shape[1]} bits)")


def test_default_view_pristine_gpu():
    """The default bge-large view of the pristine abcd.py (logs/pre_R19) and of the new one are byte-identical."""
    spec = importlib.util.spec_from_file_location("abcd_pristine", ROOT / "logs" / "pre_R19" / "abcd.py")
    old = importlib.util.module_from_spec(spec)
    sys.modules["abcd_pristine"] = old  # dataclasses look the module up there
    spec.loader.exec_module(old)
    for ds in ("star", "abcd"):
        tr, blist = load_examples("train", ds)
        dv, _ = load_examples("dev", ds)
        E = np.load(ROOT / "data" / "emb" / f"{ds}__bge-large.npz")
        em = {s: {f: E[f"{s}__{f}"] for f in ("cust", "recent", "full")} for s in ("train", "dev")}
        parts = ("mem", "intent", "step", "words")
        a = old.ABCDView(parts=parts, seed=0).fit(tr, em["train"], blist)
        b = ABCDView(parts=parts, seed=0).fit(tr, em["train"], blist)
        assert a.names == b.names
        for s, rows in (("train", tr), ("dev", dv)):
            xa, xb = a.transform(rows, em[s]), b.transform(rows, em[s])
            assert xa.tobytes() == xb.tobytes(), (ds, s)
            log(f"default view {ds} {s}: pristine == new, sha256 {hashlib.sha256(xb.tobytes()).hexdigest()[:16]}")


if __name__ == "__main__":
    test_table_and_caches()
    test_view_blocks()
    test_exists_or_equals_max()
    if os.environ.get("DL_TEST_GPU") == "1":
        test_default_view_pristine_gpu()
    print("ok")
