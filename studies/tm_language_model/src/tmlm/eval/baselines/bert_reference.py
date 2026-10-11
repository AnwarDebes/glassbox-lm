"""BERT-base MLM as a CONTAMINATED REFERENCE (it saw Wikipedia; docs/06 "Baselines").
Runs in .venv-dl (torch, transformers). Validation only in M2.

BERT has its own WordPiece vocabulary, so its top-1 on the BPE-8000 subword targets is not
defined. The honest comparison is on the whole-word frozen set: for every hidden whole word
(a maximal run of non-whitespace characters of the raw text, `tmlm.data.bpe` word ids) BERT
sees the raw text with EVERY hidden word of the window replaced by [MASK] tokens (one per
WordPiece of the word, all hidden words masked at once as in the frozen set), predicts each
masked piece independently (the standard parallel MLM readout, no left-to-right decoding), and
the pieces are joined ("##" stripped) into a predicted word string. A word is right when the
predicted string equals the gold word after the tokenizer's own normalisation (lower-casing and
accent stripping for the uncased model, whitespace removed on both sides). Words whose
WordPieces contain [UNK] can never be right and are counted as wrong.

Windows: `window` WordPieces including [CLS] and [SEP], stride `stride` over each document; a
word is scored from the window whose start is the multiple of `stride` that puts the word's
first piece closest to the window centre. Every number this module reports carries the label
"contaminated reference (saw Wikipedia)" in its note.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import torch

from tmlm.data.corpus_store import load_raw_split

CONTAMINATED = "contaminated reference (saw Wikipedia): BERT-base was pre-trained on English Wikipedia"


def _norm(tokenizer, s: str) -> str:
    try:
        s = tokenizer.backend_tokenizer.normalizer.normalize_str(s)
    except Exception:  # noqa: BLE001
        s = s.lower()
    return re.sub(r"\s+", "", s)


@torch.no_grad()
def run(ev, metric, out_dir: Path, *, model_name: str = "bert-base-uncased", window: int = 512, stride: int = 256,
        batch: int = 32, limit: int | None = None, raw_root=None) -> dict:
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    from .transformer_small import NvmlPowerSampler

    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device).eval()
    metric("bert.model", {"name": model_name, "n_params": sum(p.numel() for p in model.parameters()),
                          "vocab": tokenizer.vocab_size, "device": str(device)}, note=CONTAMINATED)
    raw = load_raw_split("wikitext103", "validation", root=raw_root)
    hidden_words = set(int(x) for x in ev.ww["hidden_words"])
    n_hidden_words = len(hidden_words)
    if limit is not None:
        hidden_words = set(sorted(hidden_words)[:limit])
    mask_id, cls_id, sep_id, unk_id = tokenizer.mask_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.unk_token_id
    inner = window - 2
    # ---------------------------------------------------------------- build windows
    jobs: list[dict] = []          # one per window: ids (with masks), doc, start, words [(gid, piece idx list, gold)]
    word_counter = 0
    n_unk_words, n_skipped = 0, 0
    sampler = NvmlPowerSampler()
    sampler.start()
    for d in range(raw.n_doc):
        text = raw.article(d)
        words = [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"\S+", text)]
        gids = range(word_counter, word_counter + len(words))
        word_counter += len(words)
        mine = [(g, words[g - gids.start]) for g in gids if g in hidden_words]
        if not mine:
            continue
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, return_attention_mask=False,
                        return_token_type_ids=False)
        ids = np.array(enc["input_ids"], dtype=np.int64)
        offs = np.array(enc["offset_mapping"], dtype=np.int64).reshape(-1, 2)
        piece_masked = np.zeros(ids.shape[0], dtype=bool)
        word_pieces: list[tuple[int, np.ndarray, str]] = []
        for g, (s, e, w) in mine:
            pieces = np.flatnonzero((offs[:, 0] < e) & (offs[:, 1] > s))
            if pieces.shape[0] == 0 or pieces.shape[0] > inner:
                n_skipped += 1
                continue
            if np.any(ids[pieces] == unk_id):
                n_unk_words += 1
            piece_masked[pieces] = True
            word_pieces.append((g, pieces, w))
        n_pieces = ids.shape[0]
        by_window: dict[int, list] = {}
        for g, pieces, w in word_pieces:
            centre = int(pieces[0]) - inner // 2
            start = int(round(centre / stride)) * stride
            start = max(0, min(start, max(0, n_pieces - inner)))
            if pieces[-1] >= start + inner:
                start = max(0, min(int(pieces[0]), max(0, n_pieces - inner)))
            by_window.setdefault(start, []).append((g, pieces, w))
        for start, ws in by_window.items():
            seg = ids[start:start + inner].copy()
            seg[piece_masked[start:start + inner]] = mask_id
            jobs.append({"ids": np.concatenate([[cls_id], seg, [sep_id]]), "start": start, "words": ws})
    build_s = time.time() - t_start
    metric("bert.windows", {"n_windows": len(jobs), "window": window, "stride": stride, "build_wall_s": round(build_s, 1)})
    metric("bert.n_hidden_words", n_hidden_words, note="hidden words of the frozen whole-word validation set")
    metric("bert.n_words_scored_limit", limit if limit is not None else n_hidden_words)
    metric("bert.n_words_skipped", n_skipped, note="no WordPiece overlap or longer than a window; counted as wrong")
    metric("bert.n_words_with_unk_piece", n_unk_words, note="cannot be predicted; counted as wrong")
    # ---------------------------------------------------------------- score
    jobs.sort(key=lambda j: j["ids"].shape[0])
    correct: dict[int, bool] = {}
    n_single, n_single_ok = 0, 0
    t0 = time.time()
    for i in range(0, len(jobs), batch):
        group = jobs[i:i + batch]
        L = max(j["ids"].shape[0] for j in group)
        x = np.full((len(group), L), tokenizer.pad_token_id, dtype=np.int64)
        att = np.zeros((len(group), L), dtype=np.int64)
        for r, j in enumerate(group):
            x[r, : j["ids"].shape[0]] = j["ids"]
            att[r, : j["ids"].shape[0]] = 1
        logits = model(input_ids=torch.from_numpy(x).to(device), attention_mask=torch.from_numpy(att).to(device)).logits
        pred = logits.argmax(-1).cpu().numpy()
        for r, j in enumerate(group):
            for g, pieces, w in j["words"]:
                pp = pred[r, pieces - j["start"] + 1]
                toks = tokenizer.convert_ids_to_tokens([int(t) for t in pp])
                pred_word = _norm(tokenizer, "".join(t[2:] if t.startswith("##") else t for t in toks))
                ok = pred_word == _norm(tokenizer, w)
                correct[g] = ok
                if pieces.shape[0] == 1:
                    n_single += 1
                    n_single_ok += int(ok)
    energy = sampler.stop()
    score_s = time.time() - t0
    n_target = len(hidden_words)
    n_ok = sum(1 for g in hidden_words if correct.get(g, False))
    acc = n_ok / max(1, n_target)
    metric("bert.validation_wholeword.word_acc", acc, note=CONTAMINATED + "; predicted word string == gold word string after normalisation")
    metric("bert.validation_wholeword.n_words", n_target)
    metric("bert.validation_wholeword.single_piece_fraction", n_single / max(1, n_target), note="hidden words that are one WordPiece")
    metric("bert.validation_wholeword.single_piece_word_acc", n_single_ok / max(1, n_single), note=CONTAMINATED)
    metric("bert.validation_wholeword.multi_piece_word_acc", (n_ok - n_single_ok) / max(1, n_target - n_single), note=CONTAMINATED)
    metric("bert.score_wall_s", round(score_s, 1), unit="s", note="shared GPU lock: indicative only")
    metric("bert.joules_raw", energy.get("joules"), note="NVML during window building and scoring, idle not subtracted, shared lock")
    metric("bert.total_wall_s", round(time.time() - t_start, 1), unit="s")
    return {"word_acc": acc, "n_words": n_target}
