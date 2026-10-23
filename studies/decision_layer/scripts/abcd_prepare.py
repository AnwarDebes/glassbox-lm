#!/usr/bin/env python
"""ABCD next-action examples and their frozen embeddings.

1. One example per action turn (src/dl/abcd.py), all three splits.
2. ``ast_count``: how many examples the paper's AST code (asappresearch/abcd,
   utils/process.py, ASTProcessor) makes from that action turn, with its tokenizer
   (bert-base-uncased or roberta-base, plus the non-enumerable slot tokens): 3 positions for
   verify-identity and validate-purchase, 1 otherwise, each kept only when ``value_to_id``
   finds the value. The paper's B-Slot accuracy is then the accuracy weighted by this count.
3. Embeddings of the text fields ``cust``, ``recent``, ``full`` (each unique text once).

Writes data/abcd/examples_<split>.json.gz and data/emb/abcd__<encoder>.npz (keys
"<split>__<field>"). No model is trained or scored here.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(1, str(ROOT / "scripts"))

from dl.abcd import ABCD_DIR, SPLITS, action_examples, buttons, load_convs, ontology, sources  # noqa: E402

FIELDS = ("cust", "recent", "full")


class ASTCounter:
    """The paper's example filter (ASTProcessor.collect_examples / value_to_id), verbatim logic."""

    def __init__(self, ont: dict, kind: str):
        from transformers import BertTokenizer, RobertaTokenizer

        special = [f"<{slot}>" for _cat, slots in ont["values"]["non_enumerable"].items() for slot in slots]
        tok = BertTokenizer.from_pretrained("bert-base-uncased") if kind == "bert" else RobertaTokenizer.from_pretrained("roberta-base")
        tok.add_tokens(special)
        self.tok = tok
        self.enumerable = {cat: [v.lower() for v in vals] for cat, vals in ont["values"]["enumerable"].items()}
        value_list = []
        for _cat, vals in ont["values"]["enumerable"].items():
            for v in vals:
                if v not in value_list:
                    value_list.append(v.lower())
        self.value_ids = {v: i for i, v in enumerate(value_list)}
        self.start_idx = len(self.value_ids)
        self.value_by_action = {a: t for _s, acts in ont["actions"].items() for a, t in acts.items()}
        self.cache: dict[str, list[str]] = {}

    def _tok(self, text: str) -> list[str]:
        if text not in self.cache:
            self.cache[text] = self.tok.tokenize(text)
        return self.cache[text]

    def value_to_id(self, context: list[str], action: str, value: str, potential_vals: list[str]) -> int:
        target_id = -1
        action_tokens = self._tok(action)
        filtered: list[str] = []
        seen: set[str] = set()
        for utterance in context:
            _speaker, text = utterance.split("|")
            for t in self._tok(text):
                if t in seen:
                    continue
                if len(t) > 2:
                    filtered.append(t)
                    seen.add(t)
        effective_max = 100 - (len(action_tokens) + 3)
        tokens = filtered[-effective_max:]
        for option in potential_vals:
            if option in self.enumerable:
                if value in self.enumerable[option]:
                    target_id = self.value_ids[value]
            else:
                entity = f"<{option}>"
                if entity in tokens:
                    target_id = self.start_idx + tokens.index(entity)
            if target_id >= 0:
                break
        return target_id

    def count(self, context: list[str], action: str, values: list) -> int:
        pv = self.value_by_action[action]
        if len(pv) == 0:
            return 1
        if action in ("verify-identity", "validate-purchase"):
            return sum(self.value_to_id(context, f"{action} {pos}", v, pv) >= 0 for pos, v in zip("abc", values))
        return int(self.value_to_id(context, action, values[0], pv) >= 0)


def ast_counts(convs: list, counter: ASTCounter) -> list[int]:
    """Same walk as ASTProcessor.build_features: context = 'speaker|text' utterances and 'action|button' entries."""
    out = []
    for conv in convs:
        so_far: list[str] = []
        for turn in conv["delexed"]:
            sp, utt = turn["speaker"], turn["text"]
            _, _, action, values, _ = turn["targets"]
            if sp in ("agent", "customer"):
                so_far.append(f"{sp}|{utt}")
            else:
                out.append(counter.count(so_far.copy(), action, values))
                so_far.append(f"action|{action}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", default=["bge-large"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--skip-ast", action="store_true")
    args = ap.parse_args()

    out_dir = ROOT / "data" / "abcd"
    out_dir.mkdir(parents=True, exist_ok=True)
    ont = ontology()
    blist = buttons(ont)
    convs = load_convs()
    ex = {s: action_examples(convs[s]) for s in SPLITS}
    counters = {} if args.skip_ast else {k: ASTCounter(ont, k) for k in ("bert", "roberta")}
    for s in SPLITS:
        p = out_dir / f"examples_{s}.json.gz"
        if p.exists():
            print("exists", p)
            continue
        cnt = {k: ast_counts(convs[s], c) for k, c in counters.items()}
        rows = []
        for i, e in enumerate(ex[s]):
            r = asdict(e)
            r.pop("utts")
            r["y"] = blist.index(e.button)
            for k in cnt:
                r[f"ast_count_{k}"] = cnt[k][i]
            rows.append(r)
        with gzip.open(p, "wt") as f:
            json.dump({"buttons": blist, "sources": sources(), "rows": rows}, f)
        print(s, len(rows), "examples written")

    import torch
    from sentence_transformers import SentenceTransformer

    from embed import ENCODERS  # scripts/embed.py

    for enc in args.encoders:
        path = ROOT / "data" / "emb" / f"abcd__{enc}.npz"
        if path.exists():
            print("exists", path)
            continue
        hf_id, prefix, kw = ENCODERS[enc]
        model = SentenceTransformer(hf_id, device="cuda", **kw)
        model.eval()
        arrays = {}
        t0 = time.time()
        for s in SPLITS:
            for fld in FIELDS:
                texts = [getattr(e, fld) or "(nothing yet)" for e in ex[s]]
                uniq = sorted(set(texts))
                pos = {t: i for i, t in enumerate(uniq)}
                with torch.inference_mode():
                    E = model.encode([prefix + t for t in uniq], batch_size=args.batch_size, convert_to_numpy=True,
                                     normalize_embeddings=True, show_progress_bar=False)
                arrays[f"{s}__{fld}"] = E[[pos[t] for t in texts]].astype(np.float32)
                print(enc, s, fld, len(texts), "texts,", len(uniq), "unique", f"{time.time() - t0:.0f}s", flush=True)
        prov = {"encoder": enc, "hf_id": hf_id, "prefix": prefix, "max_seq_length": int(model.max_seq_length),
                "fields": FIELDS, "normalized": True, "encode_seconds": round(time.time() - t0, 1), "sources": sources()}
        np.savez(path, provenance=json.dumps(prov), **arrays)
        print(json.dumps(prov))
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
