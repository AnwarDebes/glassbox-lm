#!/usr/bin/env python
"""Step 3a, the teacher (tu/PLAN_3a.md "Inputs"): bge-base-en-v1.5 sentence embeddings of the 3a sentences.

    python -I a3_teacher.py --cpu-check   # CPU only: fp32 embeddings of 64 fixed sentences -> tu/a3/teacher_cpu64.npy
    python -I a3_teacher.py               # the GPU job: the checks, then tu/a3/T_held.f16.npy and T_train.f16.npy

Input sentences: tu/a3/c4_sent_{held,train}_f.txt (the sampled sentences after the MultiWOZ filter, a3_mwoz_filter.py).

Teacher: data_untrusted/bge-base-en-v1.5/a5beb1e3 (MIT), every file checked against its FILES.sha256 first. It is
loaded with transformers' BertModel from the local safetensors only (local_files_only, no remote code, no pooler).
Tokens: the model's tokenizer.json (lowercasing BERT WordPiece, [CLS] ... [SEP]). A sentence over 128 tokens is cut to
its first 126 pieces plus [SEP]; the share that is cut is reported. The embedding, as the model card defines it: the
[CLS] vector of the last layer, L2-normalised, with no instruction prefix. The GPU job runs fp16 and stops unless its
embeddings of the 64 check sentences agree with the CPU fp32 file within 0.01 in cosine (min cosine >= 0.99). Outputs
are float16, in the order of the sentence files; refuses to overwrite.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent  # glassbox-lm/studies/text_understanding/tu
EMB = HERE.parent
A3 = HERE / "a3"
D = EMB / "data_untrusted" / "bge-base-en-v1.5" / "a5beb1e3"
MAXLEN = 128
# 64 check sentences: fixed strings written for this check; the first 6 pairs are the sanity pairs. They are a numerics
# check only and never entered the table fit. The version that ran for 3a (7539e26231b77a17, kept as
# tu/archive/a3_teacher_7539e26231b77a17.py) had three evaluation texts among them, found by M1's step 0
# (tu/m1/overlap_m1.json): "what is the weather like tomorrow" (MASSIVE en val, STAR dev), "will it rain tomorrow"
# (MASSIVE en test) and "please turn off the lights in the kitchen" (HWU64 test). They are replaced here by sentences
# that match no evaluation text of build_corpus_c4.eval_texts() by the exact or the order-free key (checked
# 2026-10-21 13:47Z). tu/a3/teacher_cpu64.npy still holds the old sentences' embeddings; this script refuses to
# overwrite it or the teacher outputs, so the edit cannot mix with 3a's files.
PAIRS_PARA = [("how do i reset my password", "i forgot my password and need to change it"),
              ("will the sky over the hills stay clear this weekend", "is a storm expected over the hills in the next few days"),
              ("please dim the lamps along the hallway", "make the hallway lamps a little darker")]
PAIRS_OTHER = [("how do i reset my password", "the train leaves at noon"),
               ("will the sky over the hills stay clear this weekend", "my card was charged twice"),
               ("please dim the lamps along the hallway", "tell me a joke about cats")]
CHECK = [s for p in PAIRS_PARA + PAIRS_OTHER for s in p]
WORDS = (
    "music banks travel cooking football weather alarms email calendar news movies taxes trains flowers phones "
    "insurance rent coffee hotels flights jokes recipes lights timers maps books games stocks shoes cars doctors "
    "schools printers passwords cards transfers refunds pizza music2 rivers").split()
CHECK += [f"This is check sentence number {i}, about {w}." for i, w in enumerate(WORDS + WORDS)][: 64 - len(CHECK)]
assert len(CHECK) == 64


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verify_files() -> None:
    for line in (D / "FILES.sha256").read_text().splitlines():
        if not line.strip():
            continue
        h, name = line.split(None, 1)
        name = name.strip().lstrip("*")
        assert sha256_file(D / name) == h, f"{name} does not match FILES.sha256"


def load(device: str, dtype):
    from tokenizers import Tokenizer
    from transformers import BertModel

    tok = Tokenizer.from_file(str(D / "tokenizer.json"))
    tok.no_truncation()
    tok.no_padding()
    model, info = BertModel.from_pretrained(str(D), local_files_only=True, trust_remote_code=False, add_pooling_layer=False,
                                            dtype=dtype, output_loading_info=True)
    assert not info.get("missing_keys"), f"weights missing from the checkpoint: {info['missing_keys']}"
    model.eval().to(device)
    return tok, model


def encode(tok, model, texts, device, batch=512, report_every=0):
    enc = tok.encode_batch(texts)
    ids = [e.ids for e in enc]
    cls_id, sep_id = ids[0][0], ids[0][-1]
    assert all(x[0] == cls_id and x[-1] == sep_id for x in ids), "every sentence must be [CLS] ... [SEP]"
    n_cut = sum(len(x) > MAXLEN for x in ids)
    ids = [x if len(x) <= MAXLEN else x[: MAXLEN - 1] + [sep_id] for x in ids]
    order = np.argsort([len(x) for x in ids], kind="stable")
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float16)
    t0 = time.time()
    with torch.inference_mode():
        for bi, st in enumerate(range(0, len(order), batch)):
            idx = order[st : st + batch]
            L = max(len(ids[i]) for i in idx)
            x = torch.zeros((len(idx), L), dtype=torch.long)
            m = torch.zeros((len(idx), L), dtype=torch.long)
            for r, i in enumerate(idx):
                x[r, : len(ids[i])] = torch.tensor(ids[i])
                m[r, : len(ids[i])] = 1
            h = model(input_ids=x.to(device), attention_mask=m.to(device)).last_hidden_state[:, 0].float()
            h = torch.nn.functional.normalize(h, dim=1)
            out[idx] = h.cpu().numpy().astype(np.float16)
            if report_every and bi % report_every == 0:
                print(f"  {st + len(idx):,} of {len(texts):,} ({time.time() - t0:.0f}s)", flush=True)
    lens = np.array([len(x) for x in ids])
    return out, {"n": len(texts), "cut_share": round(n_cut / len(texts), 5), "mean_tokens": round(float(lens.mean()), 2),
                 "max_tokens_after_cut": int(lens.max())}


def main() -> None:
    t0 = time.time()
    A3.mkdir(exist_ok=True)
    verify_files()
    if sys.argv[1:] == ["--cpu-check"]:
        out = A3 / "teacher_cpu64.npy"
        assert not out.exists(), f"{out} exists"
        torch.set_num_threads(1)
        tok, model = load("cpu", torch.float32)
        E, st = encode(tok, model, CHECK, "cpu", batch=16)
        np.save(out, E.astype(np.float32))
        para = [float(E[2 * i] @ E[2 * i + 1]) for i in range(3)]
        other = [float(E[6 + 2 * i] @ E[6 + 2 * i + 1]) for i in range(3)]
        print(json.dumps({"cpu_check": str(out), "paraphrase_cos": [round(c, 3) for c in para],
                          "unrelated_cos": [round(c, 3) for c in other], "stats": st, "seconds": round(time.time() - t0, 1)}))
        return
    assert sys.argv[1:] == [], sys.argv
    outs = {sp: A3 / f"T_{sp}.f16.npy" for sp in ("held", "train")}
    meta_p = A3 / "teacher_meta.json"
    for p in [*outs.values(), meta_p]:
        assert not p.exists(), f"{p} exists"
    ref = np.load(A3 / "teacher_cpu64.npy")
    tok, model = load("cuda", torch.float16)
    E, _ = encode(tok, model, CHECK, "cuda", batch=64)
    cos = (E.astype(np.float32) * ref).sum(1)
    assert cos.min() >= 0.99, f"fp16 GPU vs fp32 CPU: min cosine {cos.min():.4f}"
    meta = {"check_min_cos_fp16_vs_fp32": round(float(cos.min()), 5),
            "paraphrase_cos": [round(float(E[2 * i].astype(np.float32) @ E[2 * i + 1].astype(np.float32)), 3) for i in range(3)],
            "unrelated_cos": [round(float(E[6 + 2 * i].astype(np.float32) @ E[6 + 2 * i + 1].astype(np.float32)), 3) for i in range(3)]}
    print(json.dumps(meta), flush=True)
    for sp, p in outs.items():
        texts = (A3 / f"c4_sent_{sp}_f.txt").read_text(encoding="utf-8").split("\n")
        assert texts[-1] == ""
        texts = texts[:-1]
        T, st = encode(tok, model, texts, "cuda", batch=512, report_every=500)
        np.save(p, T)
        meta[sp] = {**st, "sentences_sha256": sha256_file(A3 / f"c4_sent_{sp}_f.txt")[:16], "out_sha256": sha256_file(p)[:16]}
        print(json.dumps({sp: meta[sp]}), flush=True)
    meta.update({"teacher": "bge-base-en-v1.5@a5beb1e3", "pooling": "cls, l2-normalised", "max_tokens": MAXLEN,
                 "dtype": "float16 on cuda", "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
                 "a3_teacher.py": sha256_file(Path(__file__).resolve())[:16], "seconds": round(time.time() - t0, 1),
                 "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
    meta_p.write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
