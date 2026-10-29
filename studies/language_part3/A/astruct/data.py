"""S0: MultiWOZ 2.2 system replies, delexicalised, tokenized, split; the vocabulary and the word lists.

Rules (registered in A/REGISTRATION_R1.md before any run):
- Only the train folder and dialog_acts.json are opened. dialog_acts.json holds the acts of every split in one
  JSON object; it is parsed and only the train dialogue ids are kept (no field of another entry is used).
- A reply = a SYSTEM turn whose dialog_act is not empty. Inputs come from dialog_act only.
- Split by h = SHA-256 of the UTF-8 dialogue_id as a big-endian integer, mod 100: fit h >= 12, tune-cal 0-5,
  tune-hold 6-11. Learned-arm training rows: fit dialogues with h in 12-33 (22 of the 88 fit residues).
- Delexicalisation: span_info spans sorted by (start, -end, act, slot); a span that starts before the end of the
  last kept span is skipped; a kept span becomes "[domain-slot]", domain = the act name before "-", lower case.
- Tokens: the text between kept spans goes through tokens() of the C4 table's tokenize_ref.py (never ids()); each
  placeholder is put back whole as one token; then the end token. Vocabulary: fit word types with count >= 2, every
  placeholder type seen in fit, <end>, <unk>; every other token (input or target) is <unk>.
- Expected placeholders: one per (act, slot, value) whose value (stripped, lower case) is not none, ? or empty, kept
  only when its placeholder type is in the vocabulary.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

END, UNK = "<end>", "<unk>"
FIT, CAL, HOLD = 0, 1, 2
SPLIT_NAMES = {FIT: "fit", CAL: "tune-cal", HOLD: "tune-hold"}
TRAIN_ROWS_H = (12, 33)  # inclusive residue range of the learned-arm training rows
NO_VALUE = ("none", "?", "")


def h100(dialogue_id: str) -> int:
    return int.from_bytes(hashlib.sha256(dialogue_id.encode("utf-8")).digest(), "big") % 100


def salted_mod(dialogue_id: str, salt: str, m: int) -> int:
    """A second deterministic hash for sub-samples that must not depend on h (salt names the use)."""
    return int.from_bytes(hashlib.sha256((salt + ":" + dialogue_id).encode("utf-8")).digest(), "big") % m


def split_of(h: int) -> int:
    return FIT if h >= 12 else (CAL if h <= 5 else HOLD)


def hash64(s: str) -> int:
    """Stable non-negative 63-bit id of a canonical string (condition symbols)."""
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big") & 0x7FFFFFFFFFFFFFFF


def placeholder(act: str, slot: str) -> str:
    return "[" + act.split("-")[0].lower() + "-" + slot.lower() + "]"


def is_placeholder(tok: str) -> bool:
    return tok.startswith("[") and tok.endswith("]") and len(tok) > 2


def has_value(v) -> bool:
    return str(v).strip().lower() not in NO_VALUE


def conditions(entry: dict) -> dict:
    """The inputs of one turn, from dialog_act only: domain-acts, (act, slot) pairs, expected placeholders."""
    A = entry.get("dialog_act") or {}
    acts = sorted(A)
    pairs = sorted({(a, str(s).lower()) for a in A for s, v in A[a]})
    expected = sorted(placeholder(a, s) for a in A for s, v in A[a] if has_value(v))
    return {"acts": acts, "pairs": pairs, "expected": expected,
            "S": "|".join(f"{a}:{s}" for a, s in pairs), "A": "|".join(acts)}


def kept_spans(utt: str, span_info: list) -> tuple[list, int, int]:
    """(kept spans, skipped as overlapping, skipped as an empty or out-of-range character span)."""
    spans = sorted(((int(x[3]), int(x[4]), str(x[0]), str(x[1])) for x in span_info),
                   key=lambda z: (z[0], -z[1], z[2], z[3]))
    kept, n_over, n_range, end = [], 0, 0, 0
    for s, e, a, sl in spans:
        if not (0 <= s < e <= len(utt)):
            n_range += 1
            continue
        if s < end:
            n_over += 1
            continue
        kept.append((s, e, placeholder(a, sl)))
        end = e
    return kept, n_over, n_range


def reply_tokens(utt: str, kept: list, tokens) -> list:
    out, pos = [], 0
    for s, e, ph in kept:
        out += tokens(utt[pos:s])
        out.append(ph)
        pos = e
    out += tokens(utt[pos:])
    return out


def delex_string(utt: str, kept: list) -> str:
    parts, pos = [], 0
    for s, e, ph in kept:
        parts += [utt[pos:s], ph]
        pos = e
    parts.append(utt[pos:])
    return "".join(parts)


def token_count_ok(utt: str, kept: list, toks: list, tokens) -> bool:
    """Review 17: the count equals tokens() on the raw delexicalised reply with each placeholder counted once."""
    n_raw = len(tokens(delex_string(utt, kept))) - sum(len(tokens(ph)) - 1 for _, _, ph in kept)
    return n_raw == len(toks)


def read_manifest(mwoz: Path) -> dict:
    out = {}
    for line in (mwoz / "MANIFEST.sha256").read_text().splitlines():
        if line.strip():
            h, p = line.split(None, 1)
            out[p.strip().lstrip("./")] = h
    return out


def load_train(mwoz: Path, verify: bool = True, opener=open) -> tuple[list, dict, dict]:
    """The train dialogues (file order) and their act entries. Opens train/*.json and dialog_acts.json only."""
    from .paths import sha256_file
    mwoz = Path(mwoz)
    files = sorted((mwoz / "train").glob("dialogues_*.json"))
    if not files:
        raise SystemExit(f"no train files under {mwoz}")
    man = read_manifest(mwoz) if verify else {}
    src = {}
    for f in files + [mwoz / "dialog_acts.json"]:
        rel = str(f.relative_to(mwoz))
        h = sha256_file(f)
        if verify and man.get(rel) != h:
            raise SystemExit(f"SHA-256 of {rel} does not match MANIFEST.sha256")
        src[rel] = h
    dialogues = []
    for f in files:
        with opener(f) as fh:
            dialogues += json.load(fh)
    ids = [d["dialogue_id"] for d in dialogues]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate dialogue ids in train")
    with opener(mwoz / "dialog_acts.json") as fh:
        allacts = json.load(fh)
    acts = {k: allacts[k] for k in ids if k in allacts}
    n_dropped = len(allacts) - len(acts)
    del allacts
    return dialogues, acts, {"files": src, "n_train_dialogues": len(dialogues), "n_act_entries_dropped": n_dropped}


def build_replies(dialogues: list, acts: dict, tokens) -> tuple[list, dict]:
    """One record per SYSTEM turn with a non-empty dialog_act, with every S0 self-check."""
    recs, chk = [], Counter()
    for di, d in enumerate(dialogues):
        did = d["dialogue_id"]
        h = h100(did)
        ent = acts.get(did, {})
        for t in d["turns"]:
            if t["speaker"] != "SYSTEM":
                continue
            chk["system_turns"] += 1
            e = ent.get(str(t["turn_id"]), {})
            if not e.get("dialog_act"):
                chk["system_turns_without_acts"] += 1
                continue
            cond = conditions(e)
            # review 2 (iii): the condition bits are identical when span_info is deleted
            cond2 = conditions({k: v for k, v in e.items() if k != "span_info"})
            if cond2 != cond:
                raise SystemExit(f"condition bits read span_info: {did} turn {t['turn_id']}")
            chk["span_info_test_turns"] += 1
            utt = t["utterance"]
            kept, n_over, n_range = kept_spans(utt, e.get("span_info", []))
            chk["spans_kept"] += len(kept)
            chk["spans_skipped_overlap"] += n_over
            chk["spans_skipped_range"] += n_range
            toks = reply_tokens(utt, kept, tokens)
            if not token_count_ok(utt, kept, toks, tokens):
                raise SystemExit(f"token count test failed: {did} turn {t['turn_id']}")
            chk["token_count_test_replies"] += 1
            recs.append({"dlg": di, "did": did, "turn": int(t["turn_id"]), "h": h, "split": split_of(h),
                         "toks": toks, **cond})
    return recs, dict(chk)


def build_vocab(recs: list) -> tuple[list, Counter]:
    cnt = Counter(tok for r in recs if r["split"] == FIT for tok in r["toks"])
    phs = sorted(w for w in cnt if is_placeholder(w))
    words = sorted((w for w, c in cnt.items() if c >= 2 and not is_placeholder(w)), key=lambda w: (-cnt[w], w))
    assert END not in cnt and UNK not in cnt, "special tokens must not occur in the text"
    return [END, UNK] + phs + words, cnt


def is_punct(w: str) -> bool:
    return not is_placeholder(w) and w not in (END, UNK) and not any(ch.isalnum() for ch in w)


def word_lists(recs: list, vocab: list, cnt: Counter) -> dict:
    """top32 (token count), cue40 (document frequency) and pairs64 (replies containing the pair), fit only."""
    words = [w for w in vocab if w not in (END, UNK) and not is_placeholder(w)]
    top32 = sorted(words, key=lambda w: (-cnt[w], w))[:32]
    vs = set(words)
    df, pdf = Counter(), Counter()
    for r in recs:
        if r["split"] != FIT:
            continue
        for w in set(r["toks"]):
            if w in vs:
                df[w] += 1
        for p in r["pairs"]:
            pdf[p] += 1
    cue40 = sorted(words, key=lambda w: (-df[w], w))[:40]
    pairs64 = sorted(pdf, key=lambda p: (-pdf[p], p))[:64]
    return {"top32": top32, "cue40": cue40, "pairs64": [list(p) for p in pairs64]}


def spherical_kmeans(X: np.ndarray, k: int, seed: int, iters: int = 25) -> np.ndarray:
    """Centroids (k x d, unit rows) by k-means++ start and Lloyd steps on cosine; deterministic for a seed."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if n < k:
        raise ValueError(f"{n} points for {k} concepts")
    C = np.empty((k, X.shape[1]), np.float64)
    C[0] = X[rng.integers(n)]
    d2 = np.maximum(2.0 - 2.0 * (X @ C[0]), 0.0)
    for j in range(1, k):
        p = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1.0 / n)
        C[j] = X[rng.choice(n, p=p)]
        d2 = np.minimum(d2, np.maximum(2.0 - 2.0 * (X @ C[j]), 0.0))
    lab = None
    for _ in range(iters):
        sim = X @ C.T
        new = sim.argmax(1)
        if lab is not None and (new == lab).all():
            break
        lab = new
        for j in range(k):
            m = lab == j
            if not m.any():  # empty: re-seed with the point farthest from its centroid
                far = int(sim[np.arange(n), lab].argmin())
                C[j] = X[far]
                lab[far] = j
                continue
            s = X[m].sum(0)
            C[j] = s / max(np.linalg.norm(s), 1e-12)
    return C


def concepts(vocab: list, c4_dir: Path, ks=(32, 64), seeds=(0, 1)) -> dict:
    """Nearest-concept id per vocabulary entry for each k (-1: placeholder, special or no C4 row)."""
    words = (Path(c4_dir) / "words.txt").read_text(encoding="utf-8").splitlines()
    row = {w: i for i, w in enumerate(words)}
    vecs = np.load(Path(c4_dir) / "vectors.npy", mmap_mode="r")
    idx = [i for i, w in enumerate(vocab) if w in row and not is_placeholder(w) and w not in (END, UNK)]
    X = np.asarray(vecs[[row[vocab[i]] for i in idx]], dtype=np.float64)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    out = {"n_with_row": len(idx), "n_vocab": len(vocab)}
    for k, sd in zip(ks, seeds):
        C = spherical_kmeans(X, min(k, len(idx)), sd)  # fewer words than concepts only in tiny test corpora
        lab = np.full(len(vocab), -1, np.int16)
        lab[idx] = (X @ C.T).argmax(1)
        out[f"concept{k}"] = lab
    return out


def h_ranges() -> dict:
    return {"fit": [12, 99], "tune-cal": [0, 5], "tune-hold": [6, 11], "train_rows": list(TRAIN_ROWS_H)}


def _csr(lists: list) -> tuple[np.ndarray, np.ndarray]:
    off = np.zeros(len(lists) + 1, np.int64)
    off[1:] = np.cumsum([len(x) for x in lists])
    flat = np.fromiter((v for x in lists for v in x), dtype=np.int64, count=int(off[-1]))
    return off, flat


def make_corpus(mwoz: Path, c4_dir: Path, tokens, log=print, opener=open, verify: bool = True) -> tuple[dict, dict]:
    """S0 in memory: (arrays, meta). Arrays index replies in train file order."""
    dialogues, acts, src = load_train(mwoz, verify=verify, opener=opener)
    log("train dialogues", src["n_train_dialogues"], "| act entries of other splits dropped unused:",
        src["n_act_entries_dropped"])
    recs, chk = build_replies(dialogues, acts, tokens)
    log("S0 checks", json.dumps(chk))
    vocab, cnt = build_vocab(recs)
    vid = {w: i for i, w in enumerate(vocab)}
    unk = vid[UNK]
    act_names = sorted({a for r in recs for a in r["acts"]})
    pair_names = sorted({p for r in recs for p in r["pairs"]})
    aid = {a: i for i, a in enumerate(act_names)}
    pid = {p: i for i, p in enumerate(pair_names)}
    toks = [[vid.get(w, unk) for w in r["toks"]] + [vid[END]] for r in recs]
    exp = [[vid[p] for p in r["expected"] if p in vid] for r in recs]
    n_exp_all = sum(len(r["expected"]) for r in recs)
    n_exp_kept = sum(len(x) for x in exp)
    tok_off, tok = _csr(toks)
    act_off, act_ids = _csr([[aid[a] for a in r["acts"]] for r in recs])
    pair_off, pair_ids = _csr([[pid[p] for p in r["pairs"]] for r in recs])
    exp_off, exp_ids = _csr(exp)
    S_hash = np.array([hash64("S:" + r["S"]) for r in recs], np.int64)
    A_hash = np.array([hash64("A:" + r["A"]) for r in recs], np.int64)
    for name, key, hs in (("act-slot set", "S", S_hash), ("domain-act set", "A", A_hash)):
        n_str = len({r[key] for r in recs})
        if len(np.unique(hs)) != n_str:
            raise SystemExit(f"{name} hash collision")
    split = np.array([r["split"] for r in recs], np.int8)
    h = np.array([r["h"] for r in recs], np.int16)
    train_rows = (split == FIT) & (h >= TRAIN_ROWS_H[0]) & (h <= TRAIN_ROWS_H[1])
    lists = word_lists(recs, vocab, cnt)
    con = concepts(vocab, c4_dir)
    arrays = {"r_dlg": np.array([r["dlg"] for r in recs], np.int32), "r_turn": np.array([r["turn"] for r in recs], np.int16),
              "r_h": h, "r_split": split, "r_train": train_rows, "tok_off": tok_off, "tok": tok.astype(np.int32),
              "act_off": act_off, "act_ids": act_ids.astype(np.int16), "pair_off": pair_off,
              "pair_ids": pair_ids.astype(np.int16), "exp_off": exp_off, "exp_ids": exp_ids.astype(np.int32),
              "S_hash": S_hash, "A_hash": A_hash, "concept32": con["concept32"], "concept64": con["concept64"]}
    n_unk_tokens = int(sum(1 for r in recs for w in r["toks"] if w not in vid))
    fit = split == FIT
    meta = {"vocab": vocab, "act_names": act_names, "pair_names": [list(p) for p in pair_names],
            "dialogue_ids": [d["dialogue_id"] for d in dialogues], "lists": lists,
            "source": src, "checks": chk, "h_ranges": h_ranges(),
            "counts": {"replies": len(recs), "fit_replies": int(fit.sum()), "tune_cal_replies": int((split == CAL).sum()),
                       "tune_hold_replies": int((split == HOLD).sum()), "train_row_replies": int(train_rows.sum()),
                       "fit_tokens_with_end": int(sum(len(t) for t, s in zip(toks, split) if s == FIT)),
                       "vocab": len(vocab), "placeholder_types": sum(is_placeholder(w) for w in vocab),
                       "act_names": len(act_names), "pairs": len(pair_names),
                       "act_slot_sets_fit": len(np.unique(S_hash[fit])),
                       "expected_placeholders": n_exp_all, "expected_placeholders_in_vocab": n_exp_kept,
                       "tokens_mapped_to_unk": n_unk_tokens, "concept_rows": con["n_with_row"]}}
    return arrays, meta


class Corpus:
    """S0 arrays and meta, with per-reply accessors."""

    def __init__(self, arrays: dict, meta: dict):
        self.a, self.meta = arrays, meta
        self.vocab = meta["vocab"]
        self.V = len(self.vocab)
        self.END, self.UNK = self.vocab.index(END), self.vocab.index(UNK)
        self.BOS, self.PAD = self.V, self.V + 1
        self.is_ph = np.array([is_placeholder(w) for w in self.vocab])
        self.is_punct = np.array([is_punct(w) for w in self.vocab])
        self.ph_ids = np.flatnonzero(self.is_ph)
        self.n = len(arrays["r_dlg"])
        vid = {w: i for i, w in enumerate(self.vocab)}
        self.top32 = np.array([vid[w] for w in meta["lists"]["top32"]], np.int64)
        self.cue40 = np.array([vid[w] for w in meta["lists"]["cue40"]], np.int64)
        pid = {tuple(p): i for i, p in enumerate(meta["pair_names"])}
        self.pairs64 = np.array([pid[tuple(p)] for p in meta["lists"]["pairs64"]], np.int64)
        self.n_acts = len(meta["act_names"])

    @classmethod
    def load(cls, d: Path) -> Corpus:
        z = np.load(Path(d) / "corpus.npz")
        meta = json.loads((Path(d) / "meta.json").read_text())
        return cls({k: z[k] for k in z.files}, meta)

    def save(self, d: Path) -> None:
        np.savez_compressed(Path(d) / "corpus.npz", **self.a)
        (Path(d) / "meta.json").write_text(json.dumps(self.meta))

    def tokens(self, r: int) -> np.ndarray:
        o = self.a["tok_off"]
        return self.a["tok"][o[r]:o[r + 1]]

    def acts(self, r: int) -> np.ndarray:
        o = self.a["act_off"]
        return self.a["act_ids"][o[r]:o[r + 1]]

    def pairs(self, r: int) -> np.ndarray:
        o = self.a["pair_off"]
        return self.a["pair_ids"][o[r]:o[r + 1]]

    def expected(self, r: int) -> np.ndarray:
        o = self.a["exp_off"]
        return self.a["exp_ids"][o[r]:o[r + 1]]

    def replies(self, split: int) -> np.ndarray:
        return np.flatnonzero(self.a["r_split"] == split)

    def append_replies(self, items: list) -> Corpus:
        """A copy with extra replies (source reply r, new token ids ending with <end>) appended at index n, n+1, ...;
        each copies the turn's conditions (acts, pairs, expected placeholders, S, A) and dialogue of its source."""
        a = {k: v.copy() for k, v in self.a.items()}
        for key in ("r_dlg", "r_turn", "r_h", "r_split", "r_train", "S_hash", "A_hash"):
            a[key] = np.concatenate([a[key], np.array([self.a[key][r] for r, _ in items], dtype=a[key].dtype)])
        for field, off_key in (("tok", "tok_off"), ("act_ids", "act_off"), ("pair_ids", "pair_off"), ("exp_ids", "exp_off")):
            o = self.a[off_key]
            extra = [np.asarray(t, a[field].dtype) for _, t in items] if field == "tok" else \
                [self.a[field][o[r]:o[r + 1]] for r, _ in items]
            lens = np.array([len(x) for x in extra], np.int64)
            a[field] = np.concatenate([self.a[field]] + extra)
            a[off_key] = np.concatenate([o, o[-1] + np.cumsum(lens)])
        return Corpus(a, self.meta)
