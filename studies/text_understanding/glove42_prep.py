#!/usr/bin/env python
"""N7: parse GloVe 42B 300d (Common Crawl, 1.9M uncased words; the table Yadav et al. 2021 used) once into numpy.

    glove42_prep.py

Reads data_untrusted/glove42B/glove.42B.300d.hf.zip (the Stanford NLP group's mirror on Hugging Face,
stanfordnlp/glove at revision 1db2080b2d94def6e5b0386a523102f9d8849e9d; the Stanford download ran at 0.3 MB/s and
was stopped), member glove.42B.300d.txt, streamed and never extracted to disk, as plain text, and writes
glove.42B.300d.vectors.npy (float32 [words, 300]) and glove.42B.300d.words.json next to it, plus the sha256 of the zip
and of the txt member. A line whose vector does not have 300 numbers is skipped and counted.
"""
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np

G = Path(__file__).resolve().parent / "data_untrusted" / "glove42B"
t0 = time.time()
h = hashlib.sha256()
ZIP = G / "glove.42B.300d.hf.zip"
with open(ZIP, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 22), b""):
        h.update(chunk)
ht = hashlib.sha256()
with zipfile.ZipFile(ZIP) as z, z.open("glove.42B.300d.txt") as raw:
    for chunk in iter(lambda: raw.read(1 << 22), b""):
        ht.update(chunk)
words, rows, bad = [], [], 0
with zipfile.ZipFile(ZIP) as z, z.open("glove.42B.300d.txt") as raw:
    for line in io.TextIOWrapper(raw, encoding="utf-8", errors="replace"):  # a bad byte only alters that word
        parts = line.rstrip("\n").split(" ")
        if len(parts) != 301:
            bad += 1
            continue
        words.append(parts[0])
        rows.append(np.asarray(parts[1:], dtype=np.float32))
M = np.stack(rows)
np.save(G / "glove.42B.300d.vectors.npy", M)
(G / "glove.42B.300d.words.json").write_text(json.dumps(words))
print(json.dumps({"zip": ZIP.name, "zip_sha256": h.hexdigest(), "txt_sha256": ht.hexdigest(), "shape": list(M.shape), "skipped_lines": bad,
                  "distinct_words": len(set(words)), "seconds": round(time.time() - t0, 1)}))
