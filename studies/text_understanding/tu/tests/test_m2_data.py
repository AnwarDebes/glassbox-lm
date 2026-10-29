#!/usr/bin/env python
"""Tests for tu/m2_data.py on synthetic zips in MTOP's format (no MTOP label is read). A plain script.

    $PY -I -B tu/tests/test_m2_data.py

- load: classes, splits, the -1 label of an intent unseen in train, the unseen counts, the utterance IDs per split;
- load_standin: test := val (texts, labels and IDs), labels from text hashes, independent of the intent column, and
  en/test.txt is never opened (every zip member read is recorded);
- the pin: a str path and a Path to the pinned zip are both hash-checked (a str path skipped the check before
  2026-10-25: `zip_path != ZIP` compared a str with a Path);
- negatives: a missing zip, a row without 8 columns.
"""

from __future__ import annotations

import sys
import tempfile
import zipfile
from pathlib import Path

TU = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TU))
import m2_data as M

ok = {"pass": 0, "fail": 0}


def check(name: str, cond: bool, detail: str = "") -> None:
    ok["pass" if cond else "fail"] += 1
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f": {detail}"))


def row(i, intent, text):
    return "\t".join([str(i), intent, "", text, "dom", "en_XX", f"[{intent} ]", "{}"])


def write_zip(path: Path, tr, ev, te) -> None:
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("mtop/en/train.txt", "\n".join(tr) + "\n")
        z.writestr("mtop/en/eval.txt", "\n".join(ev) + "\n")
        z.writestr("mtop/en/test.txt", "\n".join(te) + "\n")


with tempfile.TemporaryDirectory() as td:
    zp = Path(td) / "fake.zip"
    tr = [row(1, "IN:B", "wake me up at 7"), row(2, "IN:A", "what is the weather"), row(3, "IN:B", "set an alarm")]
    ev = [row(4, "IN:A", "is it cold"), row(5, "IN:C", "call mom")]
    te = [row(6, "IN:C", "phone dad"), row(7, "IN:B", "alarm at 8"), row(8, "IN:A", "rain today")]
    write_zip(zp, tr, ev, te)
    d = M.load("mtop_en", zp)
    check("load: classes = sorted distinct train intents", d.intents == ["IN:A", "IN:B"], str(d.intents))
    check("load: texts by split, val = en/eval", d.texts["val"] == ["is it cold", "call mom"] and len(d.texts["test"]) == 3)
    check("load: labels; an intent unseen in train -> -1", list(d.labels["train"]) == [1, 0, 1] and list(d.labels["val"]) == [0, -1]
          and list(d.labels["test"]) == [-1, 1, 0], str(d.labels))
    check("load: the unseen counts are recorded", d.sources["unseen_intents"] == str({"val": 1, "test": 1}), d.sources["unseen_intents"])
    check("load: the utterance IDs (column 1) per split, in row order", getattr(d, "ids", {}) == {"train": ["1", "2", "3"],
          "val": ["4", "5"], "test": ["6", "7", "8"]}, str(getattr(d, "ids", None)))
    s = M.load_standin("mtop_en", zp)
    check("standin: test := val, labels from text hashes", s.texts["test"] == s.texts["val"] and list(s.labels["test"]) == list(s.labels["val"])
          and s.intents[0] == "standin_0" and all(0 <= x < 20 for x in s.labels["train"]))
    check("standin: the IDs of train and val; test's IDs are val's", getattr(s, "ids", {}) == {"train": ["1", "2", "3"],
          "val": ["4", "5"], "test": ["4", "5"]}, str(getattr(s, "ids", None)))
    zp2 = Path(td) / "fake2.zip"
    write_zip(zp2, [r.replace("IN:B", "IN:Z") for r in tr], ev, te)  # the same texts with other intents
    s2 = M.load_standin("mtop_en", zp2)
    check("standin: does not depend on the intent column", all(list(s.labels[k]) == list(s2.labels[k]) for k in ("train", "val")))
    read = []
    orig_read = zipfile.ZipFile.read

    def rec_read(self, name, pwd=None):
        read.append(name if isinstance(name, str) else name.filename)
        return orig_read(self, name, pwd)

    orig_open = zipfile.ZipFile.open

    def rec_open(self, name, *a, **k):
        read.append(name if isinstance(name, str) else name.filename)
        return orig_open(self, name, *a, **k)

    zipfile.ZipFile.read, zipfile.ZipFile.open = rec_read, rec_open
    try:
        M.load_standin("mtop_en", zp)
    finally:
        zipfile.ZipFile.read, zipfile.ZipFile.open = orig_read, orig_open
    check("standin: en/test.txt is never opened (members read: train and eval only)",
          "mtop/en/test.txt" not in read and "mtop/en/train.txt" in read and "mtop/en/eval.txt" in read, str(read))
    try:
        M.load("mtop_en", M.ZIP.parent / "missing.zip")
        check("negative: a missing zip fails", False)
    except (FileNotFoundError, AssertionError):
        check("negative: a missing zip fails", True)
    with zipfile.ZipFile(Path(td) / "bad.zip", "w") as z:
        z.writestr("mtop/en/train.txt", "a\tb\n")
        z.writestr("mtop/en/eval.txt", "")
        z.writestr("mtop/en/test.txt", "")
    try:
        M.load("mtop_en", Path(td) / "bad.zip")
        check("negative: a row without 8 columns fails", False)
    except AssertionError:
        check("negative: a row without 8 columns fails", True)
    # the pin: point the module's pinned path at a synthetic zip whose sha256 is not the pinned value; every spelling of
    # that path (absolute Path, str, relative Path) must be refused with the hash assertion
    import os

    fake = Path(td) / "pinned.zip"
    write_zip(fake, tr, ev, te)
    old, cwd = (M.ZIP, M.ZIP_SHA256), os.getcwd()
    M.ZIP, M.ZIP_SHA256 = fake, "0" * 64
    os.chdir(td)
    try:
        for form, p in (("absolute Path", fake), ("str", str(fake)), ("relative Path", Path("pinned.zip"))):
            for fn in ("load", "load_standin"):
                name = f"pin: {fn} with a {form} to the pinned zip and a wrong sha256 is refused"
                try:
                    getattr(M, fn)("mtop_en", p)
                    check(name, False, "loaded without the hash check")
                except AssertionError:
                    check(name, True)
                except Exception as e:  # noqa: BLE001 - any other failure is a wrong refusal
                    check(name, False, f"{type(e).__name__}: {e}")
    finally:
        M.ZIP, M.ZIP_SHA256 = old
        os.chdir(cwd)
    try:
        M.load("mtop_en", zp)
        check("pin: a synthetic zip at another path still loads (tests)", True)
    except AssertionError as e:
        check("pin: a synthetic zip at another path still loads (tests)", False, str(e))

print(f"m2_data tests: {ok['pass']} passed, {ok['fail']} failed")
sys.exit(0 if ok["fail"] == 0 else 1)
