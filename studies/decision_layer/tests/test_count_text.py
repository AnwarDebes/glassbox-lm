"""e72 pre-checks 1 to 4 (plan logs/plan_e72_count_table.md): the C4-word table path, src/dl/count_text.py.

    HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-dl/bin/python tests/test_count_text.py

CPU only; train and dev only.
0. The four files hashed by the e70 and e71 freezes (static_text.py, abcd.py, abcd_explore.py, abcd_confirm.py)
   equal e87's freeze record, the last one frozen before the rewording for publication of 2026-10-23, with
   static_text.py and abcd.py at their published values (docs/hash-table.md); every record frozen after the
   publication (the dated notes of 2026-10-25) holds today's values; and the e70 and e71 records hold the values
   before e77's one-line change of abcd_explore.py and abcd_confirm.py (2026-10-19) and before the rewording. (Check 0
   was updated on 2026-10-25: before, it compared the files with the e70 and e71 records and had failed since e77's
   change.)
1. The subclass changes only the table: ExistsTC given potion's own table equals ExistsT (info with the concept,
   threshold and code hashes, names, concepts, thresholds, codes, and the bits of transform and transform_max on
   every train and dev text), ABCD, fields cust and recent.
2. Tokenizer: count_text's port gives the reference's tokens and row ids (data/tables/c4words_S1000_v1/
   tokenize_ref.py, loaded by path) on every train and dev text of ABCD and STAR, fields cust, recent and the
   customer's last 3 utterances.
3. On the C4 table, transform (lookup + OR) equals transform_max (max cosine, then thresholds) on every train and
   dev text, both datasets, the three fields; two fits are identical.
4. No potion (in a fresh interpreter): after count_text.patch, ABCDView (mem, words, xcust, xrecent, and view C's
   mem, cwords, xcust, xcrecent) fits and transforms on ABCD with the empty "none" cache; static_text.table raises;
   neither sentence_transformers nor model2vec is imported.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dl.abcd import customer_recent, load_examples  # noqa: E402

TABLE = "c4words-S1000-v1"
FROZEN = {"src/dl/static_text.py", "src/dl/abcd.py", "scripts/abcd_explore.py", "scripts/abcd_confirm.py"}
# Check 0 (updated 2026-10-25) compares the four files with the LAST freeze record that pins them:
# - abcd_explore.py and abcd_confirm.py changed once after e70 and e71, for e77 (2026-10-19 21:26 UTC: "multiwoz" in
#   --dataset, one line each; logs/plan_e77_multiwoz_dev.md; pristine copies in logs/pre_e77/, local); every record
#   from e77 to e87 holds the new values;
# - static_text.py and abcd.py were reworded for publication on 2026-10-23 (comments and docstrings only); the records
#   up to e87 hold the value before that edit, and docs/hash-table.md maps it to the published one.
LAST = "logs/plan_e87_receipts.frozen.json"
PRE_E77 = {"scripts/abcd_explore.py": "d3c53f57f9ef1975182b6834ec7ed4cbb1f4123efff44c416aa78226058e8119",
           "scripts/abcd_confirm.py": "0ae667721923aff9eb0fd3ba6c6d529e0306bc94409bf19104db3f5607a9f82b"}
PUBLISHED = {"src/dl/static_text.py": ("ec89f28c7f96c4b21d0704588e5730fc909beaaf4391ce57c736cc4266c66c39",
                                      "e7037dc6ac570487cc23dd84125166b10fb1677fa48edd88d44dab0d675b5a2a"),
             "src/dl/abcd.py": ("a66c0c633ca227d24ec1f4b9ffcc97cf3b149c6fd6fc6e60919b9f9ed149cf63",
                               "d8cf795ac26b1f0589ebfa4177f7196a6110e3ddcf9d43ad1c111bf6b60b96dc")}


def log(*a):
    print(*a, flush=True)


def fields(rows):
    return {"cust": [r["cust"] for r in rows], "recent": [r["recent"] for r in rows], "crecent": [customer_recent(r) for r in rows]}


def test_frozen_files_untouched():
    from dl.static_text import _sha256

    last = json.loads((ROOT / LAST).read_text())["sources"]
    for f in sorted(FROZEN):
        now = _sha256(ROOT / f)
        if f in PUBLISHED:
            assert last[f] == PUBLISHED[f][0] and now == PUBLISHED[f][1], (LAST, f, now)
        else:
            assert now == last[f], (LAST, f, now)
    for rec in ("logs/plan_e70_verify.frozen.json", "logs/plan_e71_v2_amendment.frozen.json"):
        src = json.loads((ROOT / rec).read_text())["sources"]
        for f in sorted(FROZEN):
            assert src[f] == (PRE_E77[f] if f in PRE_E77 else PUBLISHED[f][0]), (rec, f)
    for f, h in PRE_E77.items():  # local copies: the e70/e71 state differs from today's file by e77's one line only
        p = ROOT / "logs" / "pre_e77" / Path(f).name
        if p.exists():
            old, new = p.read_text().splitlines(), (ROOT / f).read_text().splitlines()
            diff = [i for i, (a, b) in enumerate(zip(old, new)) if a != b]
            assert _sha256(p) == h and len(old) == len(new) and len(diff) == 1 and '"multiwoz"' in new[diff[0]], (f, diff)
    later = []
    for rec in sorted((ROOT / "logs").glob("*.frozen.json")):
        r = json.loads(rec.read_text())
        if r.get("frozen_utc", "") > "2026-10-23T12:00:00Z" and all(f in r.get("sources", {}) for f in FROZEN):
            for f in sorted(FROZEN):
                assert r["sources"][f] == _sha256(ROOT / f), (rec.name, f)
            later.append(rec.name)
    log(f"0. the four files equal {LAST} (static_text.py and abcd.py as published, docs/hash-table.md) and every later "
        f"record ({', '.join(later) or 'none'}); the e70 and e71 records hold the values before e77's one-line change")


def test_subclass_equals_existsT_on_potion():
    from dl.count_text import ExistsTC
    from dl.static_text import ExistsT, table

    tr, _ = load_examples("train", "abcd")
    dv, _ = load_examples("dev", "abcd")
    ftr, fdv = fields(tr), fields(dv)
    for fld in ("cust", "recent"):
        a = ExistsT(what=fld).fit(ftr[fld])
        b = ExistsTC(what=fld, table_obj=table()).fit(ftr[fld])
        assert a.info == b.info, fld
        assert a.names == b.names and np.array_equal(a.C, b.C) and np.array_equal(a.packed, b.packed), fld
        assert len(a.th) == len(b.th) and all(np.array_equal(p, q) for p, q in zip(a.th, b.th)), fld
        for s, texts in (("train", ftr[fld]), ("dev", fdv[fld])):
            assert a.transform(texts).tobytes() == b.transform(texts).tobytes(), (fld, s)
            assert a.transform_max(texts).tobytes() == b.transform_max(texts).tobytes(), (fld, s)
        log(f"1. ABCD {fld}: ExistsTC on potion's table == ExistsT (info, names, concepts, thresholds, codes; "
            f"bits on {len(tr)} train and {len(dv)} dev texts); {a.n_bits} bits, codes {a.info['codes_sha256'][:16]}")


def test_tokenizer_port():
    from dl.count_text import count_table, tokens
    from dl.static_text import EMPTY

    spec = importlib.util.spec_from_file_location("tokenize_ref_c4words", ROOT / "data" / "tables" / "c4words_S1000_v1" / "tokenize_ref.py")
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    tb = count_table(TABLE)
    assert tb.words == (ROOT / "data" / "tables" / "c4words_S1000_v1" / "words.txt").read_text(encoding="utf-8").splitlines()
    n = 0
    for ds in ("abcd", "star"):
        for split in ("train", "dev"):
            rows, _ = load_examples(split, ds)
            for fld, texts in fields(rows).items():
                got = tb.ids(texts)
                for t, g in zip(texts, got):
                    t = t or EMPTY
                    assert tokens(t) == ref.tokens(t) and g == ref.ids(t), (ds, split, fld, t[:80])
                n += len(texts)
    log(f"2. tokenizer port == reference (tokens and row ids) on {n:,} texts")


def test_or_equals_max_c4():
    from dl.count_text import ExistsTC

    for ds in ("abcd", "star"):
        tr, _ = load_examples("train", ds)
        dv, _ = load_examples("dev", ds)
        ftr, fdv = fields(tr), fields(dv)
        for fld in ("cust", "recent", "crecent"):
            x = ExistsTC(what=fld).fit(ftr[fld])
            for s, texts in (("train", ftr[fld]), ("dev", fdv[fld])):
                a, b = x.transform(texts), x.transform_max(texts)
                assert a.shape == b.shape == (len(texts), x.n_bits) and np.array_equal(a, b), (ds, fld, s)
            on = x.transform(fdv[fld]).mean()
            log(f"3. C4 {ds} {fld}: {x.n_bits} bits ({x.info['n_train_vocab']} train words), OR == max on all train and dev texts, "
                f"{on:.3f} of dev bits on, concepts {x.info['concepts_sha256'][:16]}, codes {x.info['codes_sha256'][:16]}")
            if ds == "star" and fld == "recent":
                y = ExistsTC(what=fld).fit(ftr[fld])
                assert np.array_equal(x.C, y.C) and all(np.array_equal(p, q) for p, q in zip(x.th, y.th))
                assert np.array_equal(x.packed, y.packed) and x.info == y.info
                log("3. C4: two fits identical (concepts, thresholds, codes)")


NO_POTION = r"""
import sys
from pathlib import Path
ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from dl import count_text, static_text
meta = count_text.patch("c4words-S1000-v1")
from dl.abcd import ABCDView, load_examples
tr, blist = load_examples("train", "abcd")
dv, _ = load_examples("dev", "abcd")
E = np.load(ROOT / "data" / "emb" / "abcd__none.npz")
etr, edv = ({f: E[f"{s}__{f}"] for f in ("cust", "recent", "full")} for s in ("train", "dev"))
for parts in (("mem", "words", "xcust", "xrecent"), ("mem", "cwords", "xcust", "xcrecent")):
    v = ABCDView(parts=parts, seed=0).fit(tr, etr, blist)
    X = v.transform(dv, edv)
    info = v.static_info()
    assert all(i["table"] == "c4words-S1000-v1" for i in info.values()), info
    print("4.", parts, X.shape, {k: i["n_bits"] for k, i in info.items()}, flush=True)
try:
    static_text.table()
    raise AssertionError("potion's table loaded")
except RuntimeError:
    pass
bad = [m for m in ("sentence_transformers", "model2vec", "transformers") if m in sys.modules]
assert not bad, bad
print("4. no potion: table() raises; not imported: sentence_transformers, model2vec, transformers", flush=True)
"""


def test_no_potion_subprocess():
    r = subprocess.run([sys.executable, "-c", NO_POTION, str(ROOT)], capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    assert r.returncode == 0, r.stderr[-3000:]


if __name__ == "__main__":
    test_frozen_files_untouched()
    test_tokenizer_port()
    test_or_equals_max_c4()
    test_no_potion_subprocess()
    test_subclass_equals_existsT_on_potion()
    test_frozen_files_untouched()
    print("ok")
