#!/usr/bin/env python
"""Check that export/c4words_S1000_v1/tokenize_ref.py gives exactly the harness's tokens and row ids
(glove_feats.glove_tokens + lookup in tables/ppw_S1000_d500_e0.5.npz, as w1_n7w.bits_for_words / wordtab_feats.py)
on the train and val texts of CLINC150, Banking77, HWU64 and N7's five sets (carved val). One row in
results_w1_checks.jsonl."""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, f"{_REPO_ROOT}/studies/decision_layer/src")
import glove_feats as GF  # noqa: E402
import n7_tm as N  # noqa: E402
from dl.data import load  # noqa: E402

spec = importlib.util.spec_from_file_location("ref", HERE / "export" / "c4words_S1000_v1" / "tokenize_ref.py")
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)
z = np.load(HERE / "tables" / "ppw_S1000_d500_e0.5.npz")
live = np.abs(z["codes"].astype(np.float32)).sum(1) > 0
row = {str(w): i for i, w in enumerate(z["words"]) if live[i]}
texts = []
for ds in ("clinc150", "banking77", "hwu64"):
    d = load(ds)
    for sp in ("train", "val"):
        texts += list(d.texts[sp])
for ds in N.DATASETS:
    sp, _ = N.load(ds, False)
    texts += sp["train"][0] + sp["val"][0]
bad_tok = sum(1 for t in texts if ref.tokens(t) != GF.glove_tokens(t))
bad_ids = sum(1 for t in texts if ref.ids(t) != [row[w] for w in GF.glove_tokens(t) if w in row])
rec = {"check": "tokenize_ref.py equals the harness (tokens and row ids)", "texts": len(texts),
       "token_mismatches": bad_tok, "id_mismatches": bad_ids, "passed": bad_tok == 0 and bad_ids == 0}
print(json.dumps(rec))
with open(HERE / "results_w1_checks.jsonl", "a") as fh:
    fh.write(json.dumps(rec) + "\n")
