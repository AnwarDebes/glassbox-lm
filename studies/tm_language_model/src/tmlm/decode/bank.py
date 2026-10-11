"""The exported clause bank and its evaluation on a generation window (docs/03 "Certification
spec", docs/11 sections 1, 2 and 4; interface `export_bank` of src/tmlm/common/interface.py).

A `Bank` is a read-only numpy view of a trained engine state plus the static tables a
standalone program needs to evaluate every clause: included literals per clause (CSR), sparse
votes (CSR), alive flags, the derived anchor per clause, the kind map, the stop list, the
enabled segments of the run mode, the integer class bias and the two calibration scalars.
`Bank.export()` is the dict of `export_bank()` (builtin scalars, str and numpy arrays only, so it
survives `np.savez` / `np.load` without pickling); `bank_sha256` is its canonical content hash.

`WindowEval` evaluates clauses on ONE document window (tokens plus a hidden bitmap) with the
exact literal semantics of docs/11 section 1.4 (`tm_ctx`, `tm_positive_true` in tm_core.h),
vectorised over literals. It is the Python reference behind the incremental evaluator and the
elimination decoder; `tests/test_decode.py` checks it against the C engine's `fired_clauses`.
Nothing here learns; nothing here draws random numbers.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..engine.config import (
    ANCHOR_NONE,
    CLS_EMPTY,
    INCLUDE_THRESHOLD,
    LIT_EMPTY,
    N_EXACT_SEG,
    EngineConfig,
    bag_of_seg,
    bag_range,
    derive,
    kind_of,
    n_sym,
    seg_offset,
    sym_edge,
    sym_mask,
)

BANK_FORMAT = "tmlm.bank.trackB.v1"
Q16 = 1 << 16
BIAS_SCALE = 256  # class_bias is round(256 * ln p_unigram), int32 (docs/03 "class_bias [V] i32, log-unigram")


def class_bias_from_counts(counts: np.ndarray, vocab: int) -> np.ndarray:
    """Integer log-unigram bias: round(256 * ln((n_c + 1) / (N + V))), int32 [V]. Add-one smoothing so
    that no class has -inf; computed from TRAIN counts only (docs/06 leakage rule)."""
    c = np.asarray(counts, dtype=np.float64).reshape(-1)
    if c.shape[0] != vocab:
        raise ValueError("counts must have V entries")
    logp = np.log((c + 1.0) / (c.sum() + vocab))
    return np.rint(BIAS_SCALE * logp).astype(np.int32)


def kind_of_table(vocab: int, n_pol: int) -> np.ndarray:
    """int8 [N_SYM]: kind 0..3 of every token / cluster symbol, -1 for MASK, EDGE and polarity."""
    t = np.full(n_sym(vocab, n_pol), -1, dtype=np.int8)
    for s in range(n_sym(vocab, n_pol)):
        t[s] = kind_of(vocab, s)
    return t


def _csr(rows: list[np.ndarray], dtype) -> tuple[np.ndarray, np.ndarray]:
    ptr = np.zeros(len(rows) + 1, dtype=np.int64)
    ptr[1:] = np.cumsum([r.shape[0] for r in rows])
    flat = np.concatenate(rows).astype(dtype) if rows else np.zeros(0, dtype=dtype)
    return ptr, flat


@dataclass(frozen=True)
class Bank:
    format: str
    vocab: int
    n_pol: int
    mode: str
    n_bag_k: int
    enabled_segs: np.ndarray  # int32 [n_seg], ascending
    kind_map: np.ndarray  # u16 [4, V]
    stop: np.ndarray  # u8 [V]
    lit_ptr: np.ndarray  # i64 [C + 1]
    lit_id: np.ndarray  # u32 [nnz]
    vote_ptr: np.ndarray  # i64 [C + 1]
    vote_cls: np.ndarray  # i32 [nnz]
    vote_w: np.ndarray  # i32 [nnz]
    alive: np.ndarray  # u8 [C]
    anchor: np.ndarray  # u32 [C], ANCHOR_NONE when the clause cannot fire
    class_bias: np.ndarray  # i32 [V]
    a_q16: int
    b_q16: int
    state_hash: str

    # ---------------------------------------------------------------- construction
    @staticmethod
    def from_state(state: dict[str, np.ndarray], cfg: EngineConfig, kind_map: np.ndarray, stop: np.ndarray,
                   sym_freq: np.ndarray, class_bias: np.ndarray, a_q16: int = Q16, b_q16: int = 0,
                   state_hash: str = "") -> "Bank":
        """Build the bank from a canonical engine state (docs/11 section 2). Included literal iff
        `lit_state >= 128` and the slot is used; the anchor follows section 2 (smallest `sym_freq`
        among included positive exact literals, MASK and EDGE at the maximum, ties toward the lower
        literal id)."""
        C, L = state["lit_id"].shape
        lit_id = state["lit_id"].astype(np.uint32)
        inc = (state["lit_state"] >= INCLUDE_THRESHOLD) & (lit_id != LIT_EMPTY)
        vc = state["vote_cls"].astype(np.int32)
        vw = state["vote_w"].astype(np.int32)
        used = (vc != CLS_EMPTY) & (vw != 0)
        alive = state["alive"].astype(np.uint8)
        smask, sedge = sym_mask(cfg.vocab), sym_edge(cfg.vocab)
        freq = np.asarray(sym_freq, dtype=np.uint64).copy()
        freq[smask] = np.iinfo(np.uint32).max
        freq[sedge] = np.iinfo(np.uint32).max
        lit_rows, vote_c_rows, vote_w_rows = [], [], []
        anchor = np.full(C, ANCHOR_NONE, dtype=np.uint32)
        for j in range(C):
            lits = np.sort(lit_id[j][inc[j]]) if alive[j] else np.zeros(0, dtype=np.uint32)
            lit_rows.append(lits)
            if alive[j] and lits.shape[0]:
                seg, neg = lits >> 17, lits & 1
                pos_exact = lits[(neg == 0) & (seg < N_EXACT_SEG)]
                if pos_exact.shape[0]:
                    syms = (pos_exact >> 1) & 0xFFFF
                    f = freq[syms]
                    best = np.flatnonzero(f == f.min())
                    anchor[j] = pos_exact[best].min()
            if alive[j]:
                order = np.argsort(vc[j][used[j]], kind="stable")
                vote_c_rows.append(vc[j][used[j]][order])
                vote_w_rows.append(vw[j][used[j]][order])
            else:
                vote_c_rows.append(np.zeros(0, dtype=np.int32))
                vote_w_rows.append(np.zeros(0, dtype=np.int32))
        lit_ptr, lit_flat = _csr(lit_rows, np.uint32)
        vote_ptr, vc_flat = _csr(vote_c_rows, np.int32)
        _, vw_flat = _csr(vote_w_rows, np.int32)
        return Bank(format=BANK_FORMAT, vocab=cfg.vocab, n_pol=cfg.n_pol, mode=cfg.mode, n_bag_k=cfg.n_bag_k,
                    enabled_segs=np.asarray(derive(cfg).seg_ids, dtype=np.int32),
                    kind_map=np.ascontiguousarray(kind_map, dtype=np.uint16), stop=np.ascontiguousarray(stop, dtype=np.uint8),
                    lit_ptr=lit_ptr, lit_id=lit_flat, vote_ptr=vote_ptr, vote_cls=vc_flat, vote_w=vw_flat,
                    alive=alive, anchor=anchor, class_bias=np.ascontiguousarray(class_bias, dtype=np.int32),
                    a_q16=int(a_q16), b_q16=int(b_q16), state_hash=state_hash)

    @property
    def n_clauses(self) -> int:
        return int(self.alive.shape[0])

    def literals(self, j: int) -> np.ndarray:
        return self.lit_id[self.lit_ptr[j]:self.lit_ptr[j + 1]]

    def votes(self, j: int) -> tuple[np.ndarray, np.ndarray]:
        s = slice(self.vote_ptr[j], self.vote_ptr[j + 1])
        return self.vote_cls[s], self.vote_w[s]

    def sums(self, fired: np.ndarray) -> np.ndarray:
        """int32 [V] class sums of a fired list (docs/11 section 4 step 4)."""
        out = np.zeros(self.vocab, dtype=np.int64)
        for j in np.asarray(fired, dtype=np.int64):
            c, w = self.votes(int(j))
            np.add.at(out, c, w)
        return out.astype(np.int32)

    def posting_lists(self) -> dict[int, np.ndarray]:
        """Anchor inverted index: literal id -> ascending clause ids (docs/11 section 2)."""
        idx: dict[int, list[int]] = {}
        for j in np.flatnonzero((self.alive != 0) & (self.anchor != ANCHOR_NONE)):
            idx.setdefault(int(self.anchor[j]), []).append(int(j))
        return {k: np.asarray(v, dtype=np.uint32) for k, v in idx.items()}

    # ---------------------------------------------------------------- export
    def export(self) -> dict[str, Any]:
        """The `export_bank()` dict of the model interface plus the Track B keys."""
        return {
            "format": self.format, "vocab_size": int(self.vocab), "state_hash": self.state_hash,
            "n_pol": int(self.n_pol), "mode": self.mode, "n_bag_k": int(self.n_bag_k),
            "enabled_segs": self.enabled_segs, "kind_map": self.kind_map, "stop": self.stop,
            "lit_ptr": self.lit_ptr, "lit_id": self.lit_id, "vote_ptr": self.vote_ptr,
            "vote_cls": self.vote_cls, "vote_w": self.vote_w, "alive": self.alive, "anchor": self.anchor,
            "class_bias": self.class_bias, "a_q16": int(self.a_q16), "b_q16": int(self.b_q16),
        }

    @staticmethod
    def from_export(d: dict[str, Any]) -> "Bank":
        def s(k):  # np.load gives 0-d arrays for scalars
            v = d[k]
            return v.item() if isinstance(v, np.ndarray) else v
        if str(s("format")) != BANK_FORMAT:
            raise ValueError(f"unknown bank format {s('format')!r}")
        return Bank(format=BANK_FORMAT, vocab=int(s("vocab_size")), n_pol=int(s("n_pol")), mode=str(s("mode")),
                    n_bag_k=int(s("n_bag_k")), enabled_segs=np.asarray(d["enabled_segs"], dtype=np.int32),
                    kind_map=np.asarray(d["kind_map"], dtype=np.uint16), stop=np.asarray(d["stop"], dtype=np.uint8),
                    lit_ptr=np.asarray(d["lit_ptr"], dtype=np.int64), lit_id=np.asarray(d["lit_id"], dtype=np.uint32),
                    vote_ptr=np.asarray(d["vote_ptr"], dtype=np.int64), vote_cls=np.asarray(d["vote_cls"], dtype=np.int32),
                    vote_w=np.asarray(d["vote_w"], dtype=np.int32), alive=np.asarray(d["alive"], dtype=np.uint8),
                    anchor=np.asarray(d["anchor"], dtype=np.uint32), class_bias=np.asarray(d["class_bias"], dtype=np.int32),
                    a_q16=int(s("a_q16")), b_q16=int(s("b_q16")), state_hash=str(s("state_hash")))

    def save(self, path) -> str:
        np.savez(path, **self.export())
        return bank_sha256(self.export())

    @staticmethod
    def load(path) -> "Bank":
        with np.load(path, allow_pickle=False) as z:
            return Bank.from_export({k: z[k] for k in z.files})

    def with_calibration(self, a_q16: int, b_q16: int) -> "Bank":
        d = self.export()
        d["a_q16"], d["b_q16"] = int(a_q16), int(b_q16)
        return Bank.from_export(d)


def bank_sha256(export: dict[str, Any]) -> str:
    """Canonical content hash of an exported bank: keys in sorted order, each as
    name, dtype, shape and little-endian bytes (scalars and strings as their UTF-8 repr).
    `checker_standalone.py` computes the same digest without importing this module."""
    h = hashlib.sha256()
    for k in sorted(export):
        v = export[k]
        h.update(k.encode("utf-8") + b"\0")
        if isinstance(v, np.ndarray) and v.ndim > 0:
            arr = np.ascontiguousarray(v)
            h.update(str(arr.dtype).encode() + b"\0" + str(arr.shape).encode() + b"\0")
            h.update(arr.astype(arr.dtype.newbyteorder("<"), copy=False).tobytes())
        else:
            if isinstance(v, np.ndarray):
                v = v.item()
            h.update(repr(v).encode("utf-8") + b"\0")
    return h.hexdigest()


# ---------------------------------------------------------------- window evaluation
class WindowEval:
    """Literal and clause truth on one document window `tokens` (u16 [L]) with `hidden` (u8 [L]).

    Semantics (docs/11 section 1.3 and 1.4, mirrored from tm_core.h): the target position reads
    as MASK; a hidden position reads as MASK; a position outside [0, L) reads as EDGE; an exact
    positive literal (seg, sym) is true iff the context symbol of the literal's kind at offset
    seg equals sym (MASK and EDGE compare against kind 0); a bag literal is true iff a visible,
    non-stop position in the bucket range carries the symbol; a literal on a disabled segment is
    false; a negated literal is the negation. A clause fires iff alive, it has an included
    positive exact literal (anchor) and every included literal is true.
    """

    def __init__(self, bank: Bank, tokens: np.ndarray, hidden: np.ndarray) -> None:
        self.bank = bank
        self.tokens = np.ascontiguousarray(tokens, dtype=np.uint16)
        self.hidden = np.ascontiguousarray(hidden, dtype=np.uint8)
        if self.tokens.shape != self.hidden.shape or self.tokens.ndim != 1:
            raise ValueError("tokens and hidden must be 1-d arrays of equal length")
        self.L = int(self.tokens.shape[0])
        self.smask, self.sedge = sym_mask(bank.vocab), sym_edge(bank.vocab)
        self.nsym = n_sym(bank.vocab, bank.n_pol)
        self.kind_tbl = kind_of_table(bank.vocab, bank.n_pol)
        self.enabled = np.zeros(N_EXACT_SEG + 2 * 13, dtype=bool)
        self.enabled[bank.enabled_segs] = True
        self.offsets = np.array([seg_offset(s) for s in range(N_EXACT_SEG)], dtype=np.int64)
        # The engine reads the hidden bitmap only in masked and diffusion modes (tm_hidden in
        # tm_core.h); in causal mode nothing is hidden and `hidden` is only the decoders' record of
        # the open positions (a causal decoder fills left to right, so nothing open is ever read).
        self.reads_hidden = bank.mode in ("masked", "diffusion")
        self._bag_cache: dict[tuple[int, int], np.ndarray] = {}

    def is_hidden(self, q) -> bool:
        return bool(self.hidden[q]) if self.reads_hidden else False

    # window mutation (the caches are per (position, segment); token changes invalidate them)
    def set_token(self, q: int, t: int) -> None:
        self.tokens[q] = t
        self.hidden[q] = 0
        self._bag_cache.clear()

    def clear(self, q: int) -> None:
        self.hidden[q] = 1
        self._bag_cache.clear()

    def ctx(self, p: int) -> np.ndarray:
        """u16 [16, 4] context symbols of target p, as tm_make_example builds them."""
        out = np.zeros((N_EXACT_SEG, 4), dtype=np.uint16)
        for seg in range(N_EXACT_SEG):
            if not self.enabled[seg]:
                continue
            q = p + int(self.offsets[seg])
            if q < 0 or q >= self.L:
                out[seg, :] = self.sedge
            elif q == p or self.is_hidden(q):
                out[seg, :] = self.smask
            else:
                out[seg, :] = self.bank.kind_map[:, self.tokens[q]]
        return out

    def bag_present(self, p: int, seg: int) -> np.ndarray:
        """bool [N_SYM]: symbols carried by a visible, non-stop position in the bucket of `seg` around p."""
        key = (p, seg)
        hit = self._bag_cache.get(key)
        if hit is not None:
            return hit
        present = np.zeros(self.nsym, dtype=bool)
        if self.enabled[seg]:
            k, side = bag_of_seg(seg)
            lo, hi = bag_range(k)
            first, last = (p - hi, p - lo) if side == 0 else (p + lo, p + hi)
            first, last = max(first, 0), min(last, self.L - 1)
            if first <= last:
                q = np.arange(first, last + 1)
                q = q[q != p]
                if self.reads_hidden:
                    q = q[self.hidden[q] == 0]
                q = q[self.bank.stop[self.tokens[q]] == 0]
                if q.shape[0]:
                    present[self.bank.kind_map[:, self.tokens[q]].reshape(-1)] = True
        self._bag_cache[key] = present
        return present

    def literal_truth(self, p: int, lits: np.ndarray) -> np.ndarray:
        """bool [n]: truth of literal ids `lits` for target p."""
        lits = np.asarray(lits, dtype=np.uint32)
        seg = (lits >> 17).astype(np.int64)
        sym = ((lits >> 1) & 0xFFFF).astype(np.int64)
        neg = (lits & 1).astype(bool)
        pos = np.zeros(lits.shape[0], dtype=bool)
        exact = seg < N_EXACT_SEG
        if exact.any():
            ctx = self.ctx(p)
            kind = self.kind_tbl[sym[exact]].astype(np.int64)
            special = (sym[exact] == self.smask) | (sym[exact] == self.sedge)
            kind_eff = np.where(special, 0, np.maximum(kind, 0))
            ok = (kind >= 0) | special
            pos[exact] = ok & (ctx[seg[exact], kind_eff] == sym[exact]) & self.enabled[seg[exact]]
        bag = ~exact
        if bag.any():
            idx = np.flatnonzero(bag)
            for s in np.unique(seg[idx]):
                sel = idx[seg[idx] == s]
                if s >= self.enabled.shape[0] or not self.enabled[s]:
                    continue
                present = self.bag_present(p, int(s))
                kind = self.kind_tbl[sym[sel]]
                pos[sel] = (kind >= 0) & present[sym[sel]]
        return pos != neg

    def fires(self, p: int, clauses: np.ndarray) -> np.ndarray:
        """bool [n]: whether each clause id in `clauses` fires for target p."""
        clauses = np.asarray(clauses, dtype=np.int64)
        out = np.zeros(clauses.shape[0], dtype=bool)
        if clauses.shape[0] == 0:
            return out
        b = self.bank
        cand = (b.alive[clauses] != 0) & (b.anchor[clauses] != ANCHOR_NONE)
        ids = clauses[cand]
        if ids.shape[0] == 0:
            return out
        starts, ends = b.lit_ptr[ids], b.lit_ptr[ids + 1]
        counts = ends - starts
        flat = np.concatenate([np.arange(s, e) for s, e in zip(starts, ends)]) if counts.sum() else np.zeros(0, dtype=np.int64)
        truth = self.literal_truth(p, b.lit_id[flat])
        owner = np.repeat(np.arange(ids.shape[0]), counts)
        all_true = np.ones(ids.shape[0], dtype=bool)
        np.logical_and.at(all_true, owner, truth)
        out[cand] = all_true
        return out

    def fired_all(self, p: int) -> np.ndarray:
        """u32 ascending ids of every clause of the bank that fires for target p (brute force)."""
        ids = np.arange(self.bank.n_clauses, dtype=np.int64)
        return ids[self.fires(p, ids)].astype(np.uint32)

    def scores(self, p: int) -> np.ndarray:
        return self.bank.sums(self.fired_all(p))
