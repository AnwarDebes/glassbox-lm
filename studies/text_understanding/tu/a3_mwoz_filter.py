#!/usr/bin/env python
"""Step 3a, the MultiWOZ filter (decided 19:20Z, fix 1; CPU only; before the teacher job). W1's evaluation filter
(B.eval_texts) predates MultiWOZ 2.2, which is now an evaluation set of the agent study (e77-e79, R24). So the sampled
sentences are filtered AFTER sampling by the same two rules against the MultiWOZ 2.2 dev and test utterances: a
sentence is dropped when its key equals an utterance's key, or when it shares an 8-gram of normalised words with an
utterance (B.key, B.norm, B.NGRAM of w1/build_corpus_c4.py). The utterances are read exactly as
mwoz_leak/mwoz_c4_overlap.py reads them: the "utterance" text field of every turn of
data/untrusted/multiwoz22/{dev,test}/dialogues_*.json of the decision-layer study (text only, no labels, no acts).
No resampling: the counts fall a little below 2,000,000 and 40,000 (declared). Writes tu/a3/c4_sent_{held,train}_f.txt
and tu/a3/mwoz_filter_meta.json; refuses to overwrite.
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))

import glob
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent  # glassbox-lm/studies/text_understanding/tu
EMB = HERE.parent
A3 = HERE / "a3"
sys.path.insert(0, str(EMB / "w1"))
import build_corpus_c4 as B  # noqa: E402

MW = Path(f"{_REPO_ROOT}/studies/decision_layer/data/untrusted/multiwoz22")


def sha16(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def main() -> None:
    t0 = time.time()
    meta_p = A3 / "mwoz_filter_meta.json"
    outs = {sp: A3 / f"c4_sent_{sp}_f.txt" for sp in ("held", "train")}
    for p in [meta_p, *outs.values()]:
        assert not p.exists(), f"{p} exists"
    utt, files = [], {}
    for split in ("dev", "test"):
        for f in sorted(glob.glob(str(MW / split / "dialogues_*.json"))):
            files[f"{split}/{Path(f).name}"] = sha16(Path(f))
            for dlg in json.loads(Path(f).read_text()):
                for turn in dlg["turns"]:
                    utt.append(turn["utterance"])
    keys, grams = set(), set()
    for t in utt:
        w = B.norm(t).split()
        if w:
            keys.add(B.key(t))
        for j in range(len(w) - B.NGRAM + 1):
            grams.add(" ".join(w[j : j + B.NGRAM]))
    meta = {"utterances": len(utt), "keys": len(keys), "eight_grams": len(grams), "multiwoz_files": files}
    for sp, out in outs.items():
        src = A3 / f"c4_sent_{sp}.txt"
        lines = src.read_text(encoding="utf-8").split("\n")
        assert lines[-1] == ""
        lines = lines[:-1]
        kept, n_key, n_gram, ex = [], 0, 0, []
        for s in lines:
            if B.key(s) in keys:
                n_key += 1
                ex.append(("key", s))
                continue
            w = B.norm(s).split()
            if any(" ".join(w[j : j + B.NGRAM]) in grams for j in range(len(w) - B.NGRAM + 1)):
                n_gram += 1
                ex.append(("8-gram", s))
                continue
            kept.append(s)
        out.write_text("\n".join(kept) + "\n", encoding="utf-8")
        meta[sp] = {"in": len(lines), "in_sha256": sha16(src), "dropped_key": n_key, "dropped_8gram": n_gram,
                    "kept": len(kept), "out_sha256": sha16(out), "examples": [f"{h}: {s[:200]}" for h, s in ex[:10]]}
    meta.update({"a3_mwoz_filter.py": sha16(Path(__file__).resolve()), "build_corpus_c4.py": sha16(EMB / "w1" / "build_corpus_c4.py"),
                 "seconds": round(time.time() - t0, 1), "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
    meta_p.write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "multiwoz_files"}))


if __name__ == "__main__":
    main()
