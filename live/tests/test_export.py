"""Unit tests of live/export.py: the results-page parser, the document-word slots, the tokenizer and the
book hashes. A plain script (no pytest); it exits 1 if a test fails. Run from the repository root:

    studies/tm_language_model/.venv-tm/bin/python live/tests/test_export.py

The study's virtual environment is needed because export.py imports the study's code (numpy, numba).
Nothing is fitted and no file is written outside a temporary folder.
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
import tempfile
import traceback
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIVE = HERE.parent
spec = importlib.util.spec_from_file_location("live_export", LIVE / "export.py")
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)

SYNTHETIC_PAGE = """
| id | registered claim, in short | status | qualifier | evidence (run ids) |
|---|---|---|---|---|
| H1.1 | claim one | met | | `run_a` |
| H1.2 | claim two | not run | none | none |
| H1.3 | claim three | not met (seed 1) | on seed 1 only | `run_b` |
| H17.1 | a numbers row, not a status row | +0.09 (-1.51 to +1.78) | -4.57 | at least 1.0 point |
| H17.1 | claim four | not met | none | `run_c` |
| H17.1 | a later row of the same id is ignored | met | | `run_d` |
"""


def test_table_cells() -> None:
    assert export.table_cells("| a | | `c` |") == ["a", "", "`c`"]
    assert export.table_cells("|---|---|") == ["---", "---"]
    assert export.table_cells("not a row") is None


def test_empty_qualifier_cell_stays_empty() -> None:
    ids = {r["id"]: r for r in export.parse_study_ids(SYNTHETIC_PAGE)}
    assert sorted(ids) == ["H1.1", "H1.2", "H1.3", "H17.1"], sorted(ids)
    assert ids["H1.1"] == {"id": "H1.1", "text": "claim one", "status": "met", "qualifier": ""}, ids["H1.1"]
    assert ids["H1.2"]["qualifier"] == "none"
    assert ids["H1.3"]["status"] == "not met" and ids["H1.3"]["qualifier"] == "on seed 1 only", ids["H1.3"]
    assert ids["H17.1"]["text"] == "claim four" and ids["H17.1"]["status"] == "not met", ids["H17.1"]


def test_results_page_of_the_study() -> None:
    page = export.RESULTS_PAGE
    text = page.read_text(encoding="utf-8")
    ids = export.study_ids()
    assert len(ids) == 91, len(ids)
    for r in ids:
        assert "|" not in r["qualifier"] and "|" not in r["text"], r
    # the counts agree with the count table of the page itself ("| all | ids | met | not met | not judged | not run |")
    all_row = next(export.table_cells(l) for l in text.splitlines() if l.startswith("| all |"))
    counts = [len(ids)] + [sum(r["status"] == s for r in ids) for s in export.STATUSES]
    assert counts == [int(x) for x in all_row[1:]], (counts, all_row)
    # every id whose qualifier cell is empty on the page (a status cell followed by "| |", found here with a regular
    # expression, not with table_cells) has an empty qualifier
    by_id = {r["id"]: r for r in ids}
    empty = set()
    for line in text.splitlines():
        m = re.match(r"^\| (H\d+\.\d+) \| [^|]* \| (?:met|not met|not judged|not run) \| \| ", line)
        if m and m.group(1) in by_id and m.group(1) not in empty:
            empty.add(m.group(1))
    assert "H8.2" in empty and len(empty) >= 15, sorted(empty)
    for i in empty:
        assert by_id[i]["qualifier"] == "", (i, by_id[i]["qualifier"])


def test_check_rejects_a_bad_parse() -> None:
    page = export.ROOT / "studies" / "tm_language_model" / "docs" / "results.md"
    good = export.study_ids()
    bad = [dict(r) for r in good]
    bad[0]["qualifier"] = "| `run`"
    try:
        export.check_study_ids(bad, page.read_text(encoding="utf-8"))
    except SystemExit:
        pass
    else:
        raise AssertionError("check_study_ids accepted a qualifier that starts with '|'")


def test_slots() -> None:
    # every new word of a document has its own slot until the slots run out; then the last slot is shared
    index = {"a": 0, "b": 1}
    ids, new = export.encode(["a", "x", "y", "x", "b", "z", "w"], index, 2, 3)
    assert ids == [0, 2, 3, 2, 1, 4, 4], ids
    assert new == ["x", "y", "z", "w"], new
    docs = [{"label": "d0", "new_words": ["x", "y", "z"]}, {"label": "d1", "new_words": ["x", "y", "z", "w"]}]
    assert [d["label"] for d in export.slot_overflow(docs, 3)] == ["d1"]
    # the largest number of new words in one chapter of the shipped corpus is 160 (inventory LD-02)
    assert export.N_SLOTS >= 160, export.N_SLOTS
    assert 19173 + export.N_SLOTS <= 32768


def test_tokenize_by_hand() -> None:
    cases = [
        ("It's a truth -- universally acknowledged!", ["it's", "a", "truth", "universally", "acknowledged", "!"]),
        ("Caf\u00e9 na\u00efve \u00c6sop \u0153uvre", ["cafe", "naive", "aesop", "oeuvre"]),
        ("well-known _italic_ \u201cquoted\u201d \u2018tis", ["well", "known", "italic", "quoted", "tis"]),
        ("a\u1dc4b a\u20d7b", ["ab", "ab"]),
        ("\ufb01ne \u2163 \u00bd", ["fine", "iv", "1", "2"]),
        ("", []),
    ]
    for text, want in cases:
        got = export.tokenize(text)
        assert got == want, (text, got, want)



def test_page_combining_table() -> None:
    # model.js removes after NFKD exactly the code points that unicodedata.combining() of this Python calls combining
    src = (LIVE / "model.js").read_text(encoding="utf-8")
    m = re.search(r"const COMBINING =\s*((?:'[^']*'\s*\+?\s*)+);", src)
    assert m, "no COMBINING table in model.js"
    got = set()
    for item in "".join(re.findall(r"'([^']*)'", m.group(1))).split(","):
        a, _, b = item.partition("-")
        got.update(range(int(a, 16), int(b or a, 16) + 1))
    want = {c for c in range(0x110000) if not 0xD800 <= c <= 0xDFFF and unicodedata.combining(chr(c))}
    assert got == want, (len(got ^ want), [hex(c) for c in sorted(got ^ want)[:10]])


def test_book_hash() -> None:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "pg1.txt"
        p.write_bytes(b"abc")
        assert export.sha256_file(p) == hashlib.sha256(b"abc").hexdigest()


def main() -> int:
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc(limit=3, file=sys.stdout)
    print(f"{len(tests) - failed} of {len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
