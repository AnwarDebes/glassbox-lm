"""Per-token receipts (docs/03 "Certification spec"; docs/05 E11; docs/07 G5; docs/13 H2.4).

A receipt for one generated token holds: the window (tokens and hidden flags) the prediction was
made on, the target position, the fired clauses with their included literals and their votes,
the exact class sums, the claimed token and the decision mode, the random draw that selected it
(seed, stream, counters, the 32-bit value), the temperature, the calibration scalars, the
model's `state_hash` and the sha256 of the exported bank; for the elimination decoder also the
licensing and eliminating clauses, the survivors and the conflict flag.

`checker_standalone.py` (this package; imports nothing from the project) re-derives every token
from the receipt and the bank export; `tamper()` produces the tampered variants that must fail
(docs/04 M8 ACCEPT), one per tampering mode, chosen with the counter-based RNG.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..decode.bank import Bank, bank_sha256
from ..decode.trace import Generation
from ..engine.config import ANCHOR_NONE, N_EXACT_SEG, lit_text
from ..engine.rng import rng32

RECEIPT_FORMAT = "tmlm.receipt.trackB.v1"
TAMPER_MODES = ("claimed", "drop_clause_keep_sums", "drop_clause_fix_sums", "add_clause", "vote_weight",
                "literal", "context", "state_hash", "bank_hash", "draw", "sums", "survivors")


def _clause_entry(bank: Bank, j: int, vocab: int) -> dict[str, Any]:
    lits = bank.literals(j)
    c, w = bank.votes(j)
    return {"clause": int(j), "literals": [int(x) for x in lits],
            "literal_text": [lit_text(int(x), vocab) for x in lits],
            "votes": [[int(a), int(b)] for a, b in zip(c, w)]}


def build_receipts(gen: Generation, model, bank: Bank | None = None) -> dict[str, Any]:
    """The receipt document of one generation (JSON-serialisable)."""
    bank = bank if bank is not None else model.bank()
    export = bank.export()
    cal = model.calibration
    doc: dict[str, Any] = {
        "format": RECEIPT_FORMAT, "bank_sha256": bank_sha256(export), "state_hash": model.state_hash,
        "vocab_size": int(bank.vocab), "calibration": {"a_q16": int(cal.a_q16), "b_q16": int(cal.b_q16)},
        "generation": {**gen.info, "prompt": [int(t) for t in gen.prompt], "tokens": [int(t) for t in gen.tokens],
                       "generated_positions": [int(p) for p in gen.generated_positions],
                       "engine_calls": int(gen.engine_calls)},  # no wall time here: receipts replay byte for byte
        "tokens": [],
    }
    for r in gen.records:
        sums = {str(int(c)): int(r.sums[c]) for c in np.flatnonzero(r.sums)}
        entry: dict[str, Any] = {
            "step": int(r.step), "position": int(r.position), "claimed": int(r.claimed), "mode": r.mode,
            "temperature_hex": float(r.temperature).hex(),
            "window_tokens": [int(t) for t in r.tokens], "window_hidden": [int(h) for h in r.hidden],
            "fired": [_clause_entry(bank, int(j), bank.vocab) for j in r.fired], "sums": sums,
            "draw": None if r.draw is None else {"seed": r.draw.seed, "stream": r.draw.stream, "a": r.draw.a,
                                                 "b": r.draw.b, "c": r.draw.c, "u32": r.draw.u32},
            "eliminate": None,
        }
        if r.mode == "eliminate":
            entry["eliminate"] = {"licensing": [int(j) for j in r.licensing], "eliminating": [int(j) for j in r.eliminating],
                                  "pending": int(r.pending), "survivors": [int(c) for c in np.flatnonzero(r.survivors)],
                                  "conflict": bool(r.conflict)}
        doc["tokens"].append(entry)
    return doc


def write_receipts(path: str | Path, doc: dict[str, Any]) -> Path:
    path = Path(path)
    path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_receipts(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def receipt_text(entry: dict[str, Any], token_text=None) -> str:
    """One receipt as readable lines (figure F11)."""
    name = (lambda t: repr(token_text(t))) if token_text else (lambda t: f"tok:{t}")
    lines = [f"position {entry['position']}: claimed {name(entry['claimed'])} ({entry['mode']})"]
    for cl in entry["fired"]:
        votes = ", ".join(f"{name(c)}({w:+d})" for c, w in cl["votes"])
        lines.append(f"  clause {cl['clause']}: IF " + " AND ".join(cl["literal_text"]) + f" THEN {votes}")
    top = sorted(entry["sums"].items(), key=lambda kv: (-kv[1], int(kv[0])))[:5]
    lines.append("  sums: " + ", ".join(f"{name(int(c))}={s}" for c, s in top))
    return "\n".join(lines)


# ---------------------------------------------------------------- tampering
def _pick(seed: int, mode_idx: int, tok_idx: int, n: int, salt: int = 0) -> int:
    return (rng32(seed, 99, mode_idx, tok_idx, salt) * n) >> 32


def tamper(doc: dict[str, Any], tok_idx: int, mode: str, bank: Bank, seed: int = 1) -> dict[str, Any] | None:
    """A deep copy of `doc` whose token `tok_idx` is tampered in `mode`; None when the mode does
    not apply to that token (for example dropping a clause when none fired)."""
    if mode not in TAMPER_MODES:
        raise ValueError(f"unknown tamper mode {mode}")
    d = json.loads(json.dumps(doc))
    e = d["tokens"][tok_idx]
    V = int(d["vocab_size"])
    mi = TAMPER_MODES.index(mode)
    fired = e["fired"]
    if mode == "claimed":
        e["claimed"] = (e["claimed"] + 1 + _pick(seed, mi, tok_idx, V - 1)) % V
    elif mode in ("drop_clause_keep_sums", "drop_clause_fix_sums"):
        if not fired:
            return None
        gone = fired.pop(_pick(seed, mi, tok_idx, len(fired)))
        if mode == "drop_clause_fix_sums":
            _refresh_sums(e)
    elif mode == "add_clause":
        listed = {c["clause"] for c in fired}
        alive = [int(j) for j in np.flatnonzero((bank.alive != 0) & (bank.anchor != ANCHOR_NONE)) if int(j) not in listed]
        if not alive:
            return None
        j = alive[_pick(seed, mi, tok_idx, len(alive))]
        fired.append(_clause_entry(bank, j, V))
        fired.sort(key=lambda c: c["clause"])
        _refresh_sums(e)
    elif mode == "vote_weight":
        if not fired:
            return None
        cl = fired[_pick(seed, mi, tok_idx, len(fired))]
        if not cl["votes"]:
            return None
        k = _pick(seed, mi, tok_idx, len(cl["votes"]), 1)
        cl["votes"][k][1] += 1 + _pick(seed, mi, tok_idx, 5, 2)
        _refresh_sums(e)
    elif mode == "literal":
        if not fired:
            return None
        cl = fired[_pick(seed, mi, tok_idx, len(fired))]
        k = _pick(seed, mi, tok_idx, len(cl["literals"]), 1)
        cl["literals"][k] ^= 1  # flip the polarity
    elif mode == "context":
        # make a listed clause false: change the visible token read by one of its exact token literals
        for cl in fired:
            for lit in cl["literals"]:
                seg, sym, neg = lit >> 17, (lit >> 1) & 0xFFFF, lit & 1
                if seg < N_EXACT_SEG and neg == 0 and sym < V:
                    o = seg - 8 if seg < 8 else seg - 7
                    q = e["position"] + o
                    if 0 <= q < len(e["window_tokens"]) and not e["window_hidden"][q]:
                        e["window_tokens"][q] = (e["window_tokens"][q] + 1 + _pick(seed, mi, tok_idx, V - 1)) % V
                        return d
        return None
    elif mode == "state_hash":
        d["state_hash"] = "0" * 64 if d["state_hash"] != "0" * 64 else "1" * 64
    elif mode == "bank_hash":
        d["bank_sha256"] = "0" * 64 if d["bank_sha256"] != "0" * 64 else "1" * 64
    elif mode == "draw":
        if e["draw"] is None:
            return None
        e["draw"]["u32"] = (e["draw"]["u32"] + 1 + _pick(seed, mi, tok_idx, (1 << 32) - 1)) % (1 << 32)
    elif mode == "sums":
        c = str(_pick(seed, mi, tok_idx, V))
        e["sums"][c] = e["sums"].get(c, 0) + 1
    elif mode == "survivors":
        if e["eliminate"] is None:
            return None
        s = set(e["eliminate"]["survivors"])
        if e["claimed"] in s and len(s) > 1:
            s.discard(e["claimed"])
        else:
            s.add((e["claimed"] + 1) % V)
        e["eliminate"]["survivors"] = sorted(s)
    return d


def _refresh_sums(e: dict[str, Any]) -> None:
    sums: dict[str, int] = {}
    for cl in e["fired"]:
        for c, w in cl["votes"]:
            sums[str(c)] = sums.get(str(c), 0) + w
    e["sums"] = {c: s for c, s in sums.items() if s != 0}
