"""The customer-only parts added for R20 (cwords, xcrecent in src/dl/abcd.py) and the unchanged old views.

    HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 .venv-dl/bin/python tests/test_customer_view.py
    DL_TEST_GPU=1 ... (under scripts/gpu_run.sh shared): also the default bge-large view, pristine vs edited

- The views of the pristine abcd.py (logs/pre_R20/abcd.py, equal to R19's frozen hash) and of the edited one are
  byte-identical, with equal names: view B (mem, words, xcust, xrecent) and (mem, words), train and dev, STAR and
  ABCD; with DL_TEST_GPU=1 also the default bge-large view (mem, intent, step, words).
- customer_recent() gives the customer's last 3 utterances.
- The new parts come last, in the fixed order; xcrecent's lookup + OR equals its max-cosine path on every dev text.
- The view (mem, cwords, xcust, xcrecent) reads no human-agent text: replacing every row's "recent" and "full"
  text (the turns that hold the agent's or wizard's words) leaves its bits unchanged.
Added 2026-10-18 for R20.
"""

import hashlib
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dl.abcd import PARTS, ABCDView, customer_recent, load_examples  # noqa: E402

sha = lambda X: hashlib.sha256(np.ascontiguousarray(X, dtype=np.uint8).tobytes()).hexdigest()  # noqa: E731


def log(*a):
    print(*a, flush=True)


def pristine():
    spec = importlib.util.spec_from_file_location("abcd_pre_r20", ROOT / "logs" / "pre_R20" / "abcd.py")
    old = importlib.util.module_from_spec(spec)
    sys.modules["abcd_pre_r20"] = old  # dataclasses look the module up there
    spec.loader.exec_module(old)
    return old


def test_old_views_unchanged_cpu():
    old = pristine()
    for ds in ("star", "abcd"):
        tr, blist = load_examples("train", ds)
        dv, _ = load_examples("dev", ds)
        for parts in (("mem", "words", "xcust", "xrecent"), ("mem", "words")):
            a = old.ABCDView(parts=parts, seed=0).fit(tr, {}, blist)
            b = ABCDView(parts=parts, seed=0).fit(tr, {}, blist)
            assert a.names == b.names, (ds, parts)
            for s, rows in (("train", tr), ("dev", dv)):
                xa, xb = a.transform(rows, {}), b.transform(rows, {})
                assert xa.tobytes() == xb.tobytes(), (ds, parts, s)
                log(f"old view {'+'.join(parts)} {ds} {s}: pristine == edited, sha256 {sha(xb)[:16]}")
            if parts[-1] == "xrecent":
                assert a.static_info() == b.static_info()


def test_new_parts():
    assert PARTS[-4:] == ("xcust", "xrecent", "cwords", "xcrecent")
    r = {"cust": "a | b | c | d"}
    assert customer_recent(r) == "b | c | d" and customer_recent({"cust": "x"}) == "x" and customer_recent({"cust": ""}) == ""
    for ds in ("star", "abcd"):
        tr, blist = load_examples("train", ds)
        dv, _ = load_examples("dev", ds)
        v = ABCDView(parts=("mem", "cwords", "xcust", "xcrecent"), seed=0).fit(tr, {}, blist)
        base = ABCDView(parts=("mem", "xcust"), seed=0).fit(tr, {}, blist)
        X, Xb = v.transform(dv, {}), base.transform(dv, {})
        assert np.array_equal(X[:, : Xb.shape[1]], Xb), "the new parts must come after mem and xcust"
        assert v.names[: len(base.names)] == base.names
        nw = len(v.clex.names)
        assert v.names[len(base.names)].startswith("customer's last 3 utterances contain")
        assert v.names[len(base.names) + nw].startswith("some word of the customer's last 3 utterances")
        texts = [customer_recent(r) for r in dv]
        assert np.array_equal(v.xcrecent.transform(texts), v.xcrecent.transform_max(texts)), ds
        assert set(v.static_info()) == {"xcust", "xcrecent"}
        # no human-agent text: the turns that hold the agent's (wizard's) words are replaced, the bits stay
        rows2 = [dict(r, recent="agent: zzz qqq the account was pulled up", full="zzz | qqq") for r in dv]
        assert np.array_equal(v.transform(rows2, {}), X), ds
        log(f"new parts {ds}: {X.shape[1]} bits ({nw} customer words, {v.xcrecent.n_bits} customer existsT), after mem and xcust; "
            f"OR == max on all {len(dv)} dev texts; bits unchanged when every agent/wizard text is replaced")


def test_default_view_pristine_gpu():
    old = pristine()
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
            log(f"default view {ds} {s}: pristine == edited, sha256 {sha(xb)[:16]}")


if __name__ == "__main__":
    test_new_parts()
    test_old_views_unchanged_cpu()
    if os.environ.get("DL_TEST_GPU") == "1":
        test_default_view_pristine_gpu()
    print("ok")
