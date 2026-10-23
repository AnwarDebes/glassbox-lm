#!/usr/bin/env python
"""STAR (Mosig, Mehri, Kober 2020; github.com/RasaHQ/STAR, MIT): next wizard action, in the
format of the ABCD examples (src/dl/abcd.py), so the same views and scripts apply.

- One example per labelled wizard action: a suggestion the wizard picked (its ActionLabel, e.g.
  ask_name, weather_inform_forecast) or an API query ("query:<api>"). 187 labels.
- cust = the customer's utterances so far; recent = the last 3 utterances (user and wizard text);
  prev = the wizard's earlier labelled actions in this dialogue (the memory); subflow = the
  wizard's task(s) from the scenario (used only to fit the intent centres on train).
- Splits by a hash of the DialogueID: 80% train, 10% dev, 10% test (fixed here, before any model).
- full is left empty (no whole-dialogue field: it would carry the earlier actions as text).
Writes data/star/examples_<split>.json.gz and data/emb/star__<encoder>.npz.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STAR = ROOT / "data" / "untrusted" / "star" / "STAR-master"


def split_of(dialogue_id) -> str:
    h = int(hashlib.sha256(str(dialogue_id).encode()).hexdigest(), 16) % 100
    return "train" if h < 80 else ("dev" if h < 90 else "test")


def examples() -> dict[str, list[dict]]:
    out = {"train": [], "dev": [], "test": []}
    for f in sorted(glob.glob(str(STAR / "dialogues" / "*.json"))):
        d = json.loads(Path(f).read_text())
        tasks = "+".join(sorted(c["Task"] for c in d["Scenario"]["WizardCapabilities"]))
        utts, prev = [], []
        s = split_of(d["DialogueID"])
        for e in d["Events"]:
            if e["Agent"] == "User" and e["Action"] == "utter":
                utts.append(("user", e["Text"].strip().lower()))
                continue
            if e["Agent"] != "Wizard":
                continue
            label = e["ActionLabel"] if e["Action"] == "pick_suggestion" else (f"query:{e.get('APIName', '?')}" if e["Action"] == "query" else None)
            if label is not None:
                cust = " | ".join(t for sp, t in utts if sp == "user")
                recent = " ".join(f"{sp}: {t}" for sp, t in utts[-3:])
                out[s].append({"convo_id": int(d["DialogueID"]), "k": len(prev), "flow": tasks, "subflow": tasks, "button": label,
                               "prev": list(prev), "cust": cust, "recent": recent, "full": "", "ast_count_bert": 1, "ast_count_roberta": 1})
                prev.append(label)
            if e["Action"] in ("pick_suggestion", "utter") and e.get("Text"):
                utts.append(("wizard", e["Text"].strip().lower()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", default=["bge-large"])
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    ex = examples()
    labels = sorted({r["button"] for rows in ex.values() for r in rows})
    out_dir = ROOT / "data" / "star"
    out_dir.mkdir(parents=True, exist_ok=True)
    src_sha = hashlib.sha256((ROOT / "data" / "untrusted" / "star" / "star_master.tar.gz").read_bytes()).hexdigest()
    for s, rows in ex.items():
        for r in rows:
            r["y"] = labels.index(r["button"])
        p = out_dir / f"examples_{s}.json.gz"
        if not p.exists():
            with gzip.open(p, "wt") as fh:
                json.dump({"buttons": labels, "sources": {"star_master.tar.gz": src_sha}, "rows": rows}, fh)
        print(s, len(rows), "examples,", len({r["convo_id"] for r in rows}), "dialogues")
    sys.path.insert(0, str(ROOT / "scripts"))
    import torch
    from sentence_transformers import SentenceTransformer

    from embed import ENCODERS

    for enc in args.encoders:
        path = ROOT / "data" / "emb" / f"star__{enc}.npz"
        if path.exists():
            print("exists", path)
            continue
        hf_id, prefix, kw = ENCODERS[enc]
        model = SentenceTransformer(hf_id, device="cuda", **kw)
        arrays, t0 = {}, time.time()
        for s, rows in ex.items():
            for fld in ("cust", "recent", "full"):
                texts = [r[fld] or "(nothing yet)" for r in rows]
                uniq = sorted(set(texts))
                pos = {t: i for i, t in enumerate(uniq)}
                with torch.inference_mode():
                    E = model.encode([prefix + t for t in uniq], batch_size=args.batch_size, convert_to_numpy=True,
                                     normalize_embeddings=True, show_progress_bar=False)
                arrays[f"{s}__{fld}"] = E[[pos[t] for t in texts]].astype(np.float32)
                print(enc, s, fld, len(texts), "texts,", len(uniq), "unique", f"{time.time() - t0:.0f}s", flush=True)
        np.savez(path, provenance=json.dumps({"encoder": enc, "hf_id": hf_id, "dataset": "star", "source_sha256": src_sha}), **arrays)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
