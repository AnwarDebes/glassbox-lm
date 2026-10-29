"""A synthetic corpus in the MultiWOZ 2.2 file format, for self-tests and the dry run (no real data).

Writes train/dialogues_001-002.json, dev/ and test/ decoy files (never to be opened), dialog_acts.json (with the
acts of the decoy splits too, as in the real file) and MANIFEST.sha256; and a C4 stub (words.txt, vectors.npy).
Replies are built from per-act phrases with character spans, so delexicalisation, tokens, conditions and
placeholders behave as in the real data. A few turns have no acts, one span duplicates another's range, and
some words occur once (they become <unk>).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

DOMAINS = {"Restaurant": ["name", "area", "food", "pricerange"], "Hotel": ["name", "area", "stars", "parking"],
           "Train": ["trainid", "leaveat", "destination", "day"], "Attraction": ["name", "area", "type"]}
VALUES = {"name": ["pizza hut", "the golden curry", "alpha milton", "kings college", "acorn house"],
          "area": ["north", "south", "centre", "east", "west"], "food": ["italian", "indian", "chinese", "british"],
          "pricerange": ["cheap", "moderate", "expensive"], "stars": ["3", "4", "5"], "parking": ["yes", "no"],
          "trainid": ["TR1234", "TR5678", "TR9012"], "leaveat": ["09:15", "10:30", "17:45"],
          "destination": ["cambridge", "london", "ely"], "day": ["monday", "friday", "sunday"],
          "type": ["museum", "park", "college"], "ref": ["ABC123", "XYZ789", "QQ7777"]}
PHRASE = {"name": "i recommend {v} .", "area": "it is in the {v} .", "food": "they serve {v} food .",
          "pricerange": "it is in the {v} price range .", "stars": "it has {v} stars .", "parking": "parking : {v} .",
          "trainid": "{v} leaves soon .", "leaveat": "it leaves at {v} .", "destination": "it goes to {v} .",
          "day": "it runs on {v} .", "type": "it is a {v} .", "ref": "booking was successful , the reference number is {v} ."}
REQUEST = {"name": "which name do you want ?", "area": "what area would you like ?", "food": "what type of food ?",
           "pricerange": "what price range ?", "stars": "how many stars ?", "parking": "do you need parking ?",
           "trainid": "which train ?", "leaveat": "when do you want to leave ?", "destination": "where to ?",
           "day": "what day ?", "type": "what type of place ?"}
NOISE = ["sure", "okay", "great", "certainly", "alright", "well", "so"]


def _reply(rng, dom: str):
    """(utterance, dialog_act, span_info)."""
    acts, spans, parts = {}, [], []
    pos = 0

    def put(text, slot=None, value=None, act=None):
        nonlocal pos
        if parts:
            parts.append(" ")
            pos += 1
        if slot is None:
            parts.append(text)
            pos += len(text)
            return
        pre, post = text.split("{v}")
        parts.append(pre)
        s = pos + len(pre)
        parts.append(value)
        parts.append(post)
        spans.append([act, slot, value, s, s + len(value)])
        pos += len(pre) + len(value) + len(post)

    if rng.random() < 0.4:
        put(NOISE[rng.integers(len(NOISE))] + " ,")
    kind = rng.random()
    slots = DOMAINS[dom]
    if kind < 0.45:
        k = int(rng.integers(1, 3))
        for sl in rng.choice(slots, k, replace=False):
            v = VALUES[sl][rng.integers(len(VALUES[sl]))]
            act = f"{dom}-Inform"
            acts.setdefault(act, []).append([sl, v])
            put(PHRASE[sl], sl, v, act)
            if sl == "name" and rng.random() < 0.3:  # the same mention annotated under a second act
                acts.setdefault(f"{dom}-Recommend", []).append([sl, v])
                spans.append([f"{dom}-Recommend", sl, v, spans[-1][3], spans[-1][4]])
    elif kind < 0.7:
        sl = slots[rng.integers(len(slots))]
        acts.setdefault(f"{dom}-Request", []).append([sl, "?"])
        put(REQUEST[sl])
    elif kind < 0.85:
        v = VALUES["ref"][rng.integers(3)]
        acts["Booking-Book"] = [["ref", v]]
        put(PHRASE["ref"], "ref", v, "Booking-Book")
    else:
        acts["general-reqmore"] = [["none", "none"]]
        put("is there anything else i can help you with ?")
    if rng.random() < 0.3:
        acts.setdefault("general-reqmore", [["none", "none"]])
        put("anything else ?")
    if rng.random() < 0.05:
        put(f"word{int(rng.integers(10**6))}")  # a once-only word: <unk>
    return "".join(parts), acts, spans


def make_dialogues(n: int, seed: int, prefix: str = "SYN") -> tuple[list, dict]:
    rng = np.random.default_rng(seed)
    dlgs, acts = [], {}
    for i in range(n):
        did = f"{prefix}{i:05d}.json"
        dom = list(DOMAINS)[rng.integers(len(DOMAINS))]
        turns, ent = [], {}
        for t in range(int(rng.integers(2, 5))):
            turns.append({"frames": [], "speaker": "USER", "turn_id": str(2 * t), "utterance": "i need a place ."})
            utt, a, sp = _reply(rng, dom)
            if rng.random() < 0.03:
                a, sp = {}, []
            turns.append({"frames": [], "speaker": "SYSTEM", "turn_id": str(2 * t + 1), "utterance": utt})
            ent[str(2 * t)] = {"dialog_act": {}, "span_info": []}
            ent[str(2 * t + 1)] = {"dialog_act": a, "span_info": sp}
        dlgs.append({"dialogue_id": did, "services": [dom.lower()], "turns": turns})
        acts[did] = ent
    return dlgs, acts


def write_mwoz(root: Path, n: int = 300, seed: int = 0) -> Path:
    root = Path(root)
    for sub in ("train", "dev", "test"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    dl, acts = make_dialogues(n, seed)
    half = len(dl) // 2
    (root / "train" / "dialogues_001.json").write_text(json.dumps(dl[:half]))
    (root / "train" / "dialogues_002.json").write_text(json.dumps(dl[half:]))
    dv, av = make_dialogues(20, seed + 1, prefix="DEV")
    ts, at = make_dialogues(20, seed + 2, prefix="TST")
    (root / "dev" / "dialogues_001.json").write_text(json.dumps(dv))
    (root / "test" / "dialogues_001.json").write_text(json.dumps(ts))
    (root / "dialog_acts.json").write_text(json.dumps({**acts, **av, **at}))
    lines = []
    for p in sorted(root.rglob("*.json")):
        lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  ./{p.relative_to(root)}")
    (root / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    return root


def write_c4_stub(root: Path, seed: int = 0, dim: int = 16) -> Path:
    """Every word the generator can produce (and a few more), with random unit vectors."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    words = set()
    for t in list(PHRASE.values()) + list(REQUEST.values()) + NOISE + ["is there anything else i can help you with ?",
                                                                      "anything else ?", "i need a place ."]:
        words.update(t.replace("{v}", " ").split())
    for vs in VALUES.values():
        for v in vs:
            words.update(v.lower().split())
    words = sorted(words) + [f"filler{i}" for i in range(40)]
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((len(words), dim))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    (root / "words.txt").write_text("\n".join(words) + "\n")
    np.save(root / "vectors.npy", X.astype(np.float16))
    return root
