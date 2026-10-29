#!/usr/bin/env python
"""Test 4 transformer reference: frozen all-MiniLM-L6-v2 word vectors for the MASSIVE slot words (GPU).

    minilm_slots.py --emb-dir DIR

Each utterance (utt, its words joined by spaces) goes through the frozen model once; a word's vector is the last
hidden state of its first sub-word token (the usual probe for token tagging). Writes DIR/massive_en__slots_minilm.npz
with one float16 [words, 384] array per split, in the word order of massive_slots.py (checked by the word count).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import massive_slots as MS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--emb-dir", type=Path, required=True)
args = ap.parse_args()
from transformers import AutoModel, AutoTokenizer  # noqa: E402

name = "sentence-transformers/all-MiniLM-L6-v2"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModel.from_pretrained(name).cuda().eval()
rows = [json.loads(x) for x in MS.SRC.read_text().splitlines() if x.strip()]
z = np.load(args.emb_dir / "massive_en__slots_K512.npz")
out = {}
for sp, part in MS.PART.items():
    utts = [MS.words_labels(r["annot_utt"])[0] for r in rows if r["partition"] == part]
    vecs = []
    for a in range(0, len(utts), 256):
        batch = utts[a : a + 256]
        enc = tok(batch, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.inference_mode():
            h = model(**{k: v.cuda() for k, v in enc.items()}).last_hidden_state.float().cpu().numpy()
        for i, words in enumerate(batch):
            wid = enc.word_ids(i)
            first = {}
            for t, w in enumerate(wid):
                if w is not None and w not in first:
                    first[w] = t
            assert len(first) == len(words), (sp, a + i)
            vecs.append(h[i, [first[w] for w in range(len(words))]])
    V = np.concatenate(vecs).astype(np.float16)
    assert len(V) == int(z[f"{sp}__off"][-1]), sp
    out[sp] = V
np.savez(args.emb_dir / "massive_en__slots_minilm.npz", **out)
print({sp: v.shape for sp, v in out.items()})
