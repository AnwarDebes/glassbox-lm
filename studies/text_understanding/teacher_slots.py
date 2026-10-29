#!/usr/bin/env python
"""Logic-reader step 0b (exploratory): frozen contextual word vectors of a stronger teacher for the MASSIVE slot
words, TRAIN AND VAL ONLY (the test split is not encoded). GPU.

    teacher_slots.py --emb-dir DIR --teacher bge-base|bert-base

As minilm_slots.py: each utterance (its words joined by spaces) goes through the frozen model once; a word's vector
is the last hidden state of its first sub-word token. Models from the local Hugging Face cache (pinned snapshots,
offline). Writes DIR/massive_en__slots_{teacher}.npz with one float16 [words, 768] array for train and one for val,
in the word order of massive_slots.py (checked by the word count).
"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
_HOME = _os.path.expanduser("~")
import argparse
import json
import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import massive_slots as MS  # noqa: E402

HUB = Path(f"{_HOME}/.cache/huggingface/hub")
SNAP = {"bge-base": HUB / "models--BAAI--bge-base-en-v1.5/snapshots/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
        "bert-base": HUB / "models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594"}
ap = argparse.ArgumentParser()
ap.add_argument("--emb-dir", type=Path, required=True)
ap.add_argument("--teacher", required=True, choices=sorted(SNAP))
args = ap.parse_args()
from transformers import AutoModel, AutoTokenizer  # noqa: E402

torch.cuda.set_per_process_memory_fraction(0.1)
tok = AutoTokenizer.from_pretrained(str(SNAP[args.teacher]))
model = AutoModel.from_pretrained(str(SNAP[args.teacher])).cuda().eval()
rows = [json.loads(x) for x in MS.SRC.read_text().splitlines() if x.strip()]
z = np.load(args.emb_dir / "massive_en__slots_K512.npz")
out = {}
for sp in ("train", "val"):
    part = MS.PART[sp]
    utts = [MS.words_labels(r["annot_utt"])[0] for r in rows if r["partition"] == part]
    vecs = []
    for a in range(0, len(utts), 128):
        batch = utts[a : a + 128]
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
np.savez(args.emb_dir / f"massive_en__slots_{args.teacher}.npz", **out)
print(json.dumps({"teacher": args.teacher, "snapshot": SNAP[args.teacher].name, **{sp: list(v.shape) for sp, v in out.items()}}))
