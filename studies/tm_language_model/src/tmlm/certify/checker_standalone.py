#!/usr/bin/env python3
"""Standalone receipt checker (docs/03 "Certification spec"; docs/05 E11; docs/07 G5; docs/13 H2.4).

Imports NOTHING from this project: the Python standard library and numpy only (z3 is optional,
used only with --z3 for a second, solver-based argmax check). It re-derives every generated
token from (1) the exported clause bank (`bank.npz`, the `export_bank()` dict of the model
interface) and (2) the receipt document (`receipts.json`, tmlm.certify.receipt), and fails on
any tampering:

  1. the bank's canonical sha256 and the model state_hash match the receipt;
  2. every listed clause is alive, its literal list and its votes are exactly those of the bank,
     and every literal is true on the receipt's window (literal semantics of docs/11 section 1.4,
     re-implemented here from the spec: exact offsets, MASK, EDGE, bag buckets, stop tokens);
  3. no unlisted clause of the bank fires on that window (brute force over the whole bank);
  4. the class sums are the exact integer sums of the listed votes;
  5. the claimed token follows from the sums: greedy = the integer argmax of
     a_q16 * 256 * sum + b_q16 * class_bias (ties toward the lower id); sample = the inverse-CDF
     selection with the recorded counter-based draw, whose 32-bit value is re-derived from
     (seed, stream, a, b, c) with the SplitMix64 counter RNG of docs/11 section 3; eliminate =
     the three-valued licensing / eliminating sets and the survivors are re-derived from the bank
     and the claimed token is drawn from the survivors;
  6. the decision is bound to the generation run (added 2026-10-25, finish pass, bug TLM-B2; before,
     a draw re-derived from counters that the receipt itself chose was accepted, so a receipt could
     claim another token with a forged draw, a changed temperature or a greedy mode): the counters
     of every draw are the ones the decoder must use (tmlm.decode.sampling: seed = the seed of the
     run in the receipt's generation record, stream = SAMPLER, a = the decoding step, b = the target
     position, c = DRAW_TOKEN); the decoding steps follow the decoder (one token per step for the
     any-order and elimination decoders, so the step is the token's index; for diffusion the steps
     run 0, 1, 2, ... without a gap, every token of one step shares that step's window, and the last
     step is the run's last iteration); the mode is the decoder's (sample, or greedy when the run is
     greedy; sample for diffusion; eliminate) and the temperature is the run's. With --seed and
     --temperature the run's own seed and temperature must equal the values the verifier expects
     (the registered generations), so the run record of the receipt is bound as well.

What is not checked: that the window of a token equals the prompt plus the tokens claimed before
it (the windows of consecutive tokens are not chained), and that the generation record's final
tokens equal the claims.

Usage:
    checker_standalone.py --bank bank.npz --receipts receipts.json [--seed S] [--temperature T] [--z3]
                          [--json report.json] [--quiet]
The receipts file holds one receipt document, or {"runs": [document, ...]} (the layout of the E11
run folders), in which case every document is checked. Exit status 0 iff every token passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys

import numpy as np

M64 = (1 << 64) - 1
N_EXACT_SEG = 16
CLUSTER_SIZES = (16, 256, 4096)
LIT_EMPTY = 0xFFFFFFFF
TWO32 = 1 << 32
TRUE, FALSE, UNKNOWN = 1, 0, 2
# The decoders' draws (tmlm.engine.config.Stream.SAMPLER and tmlm.decode.sampling.DRAW_TOKEN, copied
# here because this file imports nothing from the project; tests/test_certify.py checks the copies).
STREAM_SAMPLER = 15
DRAW_TOKEN = 0
DECODER_MODES = {"diffusion": ("sample",), "eliminate": ("eliminate",)}  # anyorder: from its greedy flag


# ---------------------------------------------------------------- counter-based RNG (docs/11 section 3)
def mix64(z: int) -> int:
    z = (z + 0x9E3779B97F4A7C15) & M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return z ^ (z >> 31)


def rng32(seed: int, stream: int, a: int, b: int, c: int) -> int:
    z = mix64((seed ^ ((stream * 0xD6E8FEB86659FD93) & M64)) & M64)
    z = mix64((z + a) & M64)
    z = mix64((z + b) & M64)
    return mix64((z + c) & M64) >> 32


# ---------------------------------------------------------------- the bank
def bank_sha256(export: dict) -> str:
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


def load_bank(path: str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d["_sha256"] = bank_sha256(d)
    for k in ("format", "state_hash", "mode"):
        d[k] = str(d[k].item() if isinstance(d[k], np.ndarray) else d[k])
    for k in ("vocab_size", "n_pol", "n_bag_k", "a_q16", "b_q16"):
        d[k] = int(d[k])
    V = d["vocab_size"]
    d["sym_mask"] = V + sum(CLUSTER_SIZES)
    d["sym_edge"] = d["sym_mask"] + 1
    d["n_sym"] = d["sym_edge"] + 1 + d["n_pol"]
    kind = np.full(d["n_sym"], -1, dtype=np.int8)
    kind[:V] = 0
    base = V
    for k, size in enumerate(CLUSTER_SIZES, start=1):
        kind[base:base + size] = k
        base += size
    d["kind_of"] = kind
    enabled = np.zeros(N_EXACT_SEG + 2 * 13, dtype=bool)
    enabled[np.asarray(d["enabled_segs"], dtype=np.int64)] = True
    d["enabled"] = enabled
    d["offsets"] = np.array([s - 8 if s < 8 else s - 7 for s in range(N_EXACT_SEG)], dtype=np.int64)
    C = int(d["alive"].shape[0])
    d["can_fire"] = (d["alive"] != 0) & (d["anchor"] != LIT_EMPTY)
    d["C"] = C
    return d


def clause_literals(bank: dict, j: int) -> np.ndarray:
    return bank["lit_id"][bank["lit_ptr"][j]:bank["lit_ptr"][j + 1]]


def clause_votes(bank: dict, j: int) -> tuple[np.ndarray, np.ndarray]:
    s = slice(bank["vote_ptr"][j], bank["vote_ptr"][j + 1])
    return bank["vote_cls"][s], bank["vote_w"][s]


# ---------------------------------------------------------------- literal semantics (docs/11 section 1.4)
class Window:
    def __init__(self, bank: dict, tokens: np.ndarray, hidden: np.ndarray, p: int) -> None:
        self.bank, self.tokens, self.hidden, self.p = bank, tokens, hidden, int(p)
        self.L = int(tokens.shape[0])
        self.reads_hidden = bank["mode"] in ("masked", "diffusion")  # causal: nothing is hidden (docs/11 1.3)
        self.ctx = self._ctx()
        self._bag: dict[int, np.ndarray] = {}
        self._bag_hidden: dict[int, bool] = {}

    def _ctx(self) -> np.ndarray:
        b = self.bank
        out = np.zeros((N_EXACT_SEG, 4), dtype=np.int64)
        for seg in range(N_EXACT_SEG):
            if not b["enabled"][seg]:
                continue
            q = self.p + int(b["offsets"][seg])
            if q < 0 or q >= self.L:
                out[seg, :] = b["sym_edge"]
            elif q == self.p or (self.reads_hidden and self.hidden[q]):
                out[seg, :] = b["sym_mask"]
            else:
                out[seg, :] = b["kind_map"][:, self.tokens[q]]
        return out

    def _bag_range(self, seg: int) -> tuple[int, int]:
        k, side = (seg - N_EXACT_SEG) // 2, (seg - N_EXACT_SEG) % 2
        lo, hi = 8 * (1 << k) + 1, 8 * (1 << (k + 1))
        first, last = (self.p - hi, self.p - lo) if side == 0 else (self.p + lo, self.p + hi)
        return max(first, 0), min(last, self.L - 1)

    def bag_present(self, seg: int) -> np.ndarray:
        if seg in self._bag:
            return self._bag[seg]
        b = self.bank
        present = np.zeros(b["n_sym"], dtype=bool)
        any_hidden = False
        if b["enabled"][seg]:
            first, last = self._bag_range(seg)
            if first <= last:
                q = np.arange(first, last + 1)
                q = q[q != self.p]
                any_hidden = bool(np.any(self.hidden[q]))
                if self.reads_hidden:
                    q = q[self.hidden[q] == 0]
                q = q[b["stop"][self.tokens[q]] == 0]
                if q.shape[0]:
                    present[b["kind_map"][:, self.tokens[q]].reshape(-1)] = True
        self._bag[seg], self._bag_hidden[seg] = present, any_hidden
        return present

    def truth(self, lits: np.ndarray) -> np.ndarray:
        """bool [n]: two-valued literal truth (hidden reads as MASK)."""
        b = self.bank
        lits = np.asarray(lits, dtype=np.uint32)
        seg = (lits >> 17).astype(np.int64)
        sym = ((lits >> 1) & 0xFFFF).astype(np.int64)
        neg = (lits & 1).astype(bool)
        pos = np.zeros(lits.shape[0], dtype=bool)
        exact = seg < N_EXACT_SEG
        if exact.any():
            e = np.flatnonzero(exact)
            kind = b["kind_of"][sym[e]].astype(np.int64)
            special = (sym[e] == b["sym_mask"]) | (sym[e] == b["sym_edge"])
            kind_eff = np.where(special, 0, np.maximum(kind, 0))
            pos[e] = ((kind >= 0) | special) & (self.ctx[seg[e], kind_eff] == sym[e]) & b["enabled"][seg[e]]
        for i in np.flatnonzero(~exact):
            s = int(seg[i])
            if s < b["enabled"].shape[0] and b["enabled"][s] and b["kind_of"][sym[i]] >= 0:
                pos[i] = bool(self.bag_present(s)[sym[i]])
        return pos != neg

    def truth3(self, lits: np.ndarray) -> np.ndarray:
        """int8 [n] three-valued truth (hidden token positions are UNKNOWN; MASK and EDGE literals
        read the visibility as it stands), as tmlm.decode.eliminate defines it."""
        b = self.bank
        lits = np.asarray(lits, dtype=np.uint32)
        seg = (lits >> 17).astype(np.int64)
        sym = ((lits >> 1) & 0xFFFF).astype(np.int64)
        neg = (lits & 1).astype(bool)
        two = self.truth(lits & ~np.uint32(1))
        out = np.where(two, TRUE, FALSE).astype(np.int8)
        exact = seg < N_EXACT_SEG
        if exact.any():
            e = np.flatnonzero(exact)
            q = self.p + b["offsets"][seg[e]]
            inside = (q >= 0) & (q < self.L) & (q != self.p)
            hid = np.zeros(e.shape[0], dtype=bool)
            hid[inside] = self.hidden[q[inside]] != 0
            special = (sym[e] == b["sym_mask"]) | (sym[e] == b["sym_edge"])
            out[e[hid & ~special & b["enabled"][seg[e]]]] = UNKNOWN
        for i in np.flatnonzero(~exact):
            s = int(seg[i])
            if s < b["enabled"].shape[0] and b["enabled"][s] and out[i] != TRUE:
                self.bag_present(s)
                if self._bag_hidden[s]:
                    out[i] = UNKNOWN
        flip = neg & (out != UNKNOWN)
        out[flip] = 1 - out[flip]
        return out


def all_fired(bank: dict, w: Window) -> np.ndarray:
    """Ascending ids of every clause of the bank that fires (brute force)."""
    ids = np.flatnonzero(bank["can_fire"])
    if ids.shape[0] == 0:
        return ids
    starts, ends = bank["lit_ptr"][ids], bank["lit_ptr"][ids + 1]
    counts = ends - starts
    flat = np.concatenate([np.arange(s, e) for s, e in zip(starts, ends)])
    t = w.truth(bank["lit_id"][flat])
    owner = np.repeat(np.arange(ids.shape[0]), counts)
    ok = np.ones(ids.shape[0], dtype=bool)
    np.logical_and.at(ok, owner, t)
    return ids[ok]


def all_truth3(bank: dict, w: Window) -> np.ndarray:
    C = bank["C"]
    out = np.full(C, FALSE, dtype=np.int8)
    ids = np.flatnonzero(bank["can_fire"])
    if ids.shape[0] == 0:
        return out
    starts, ends = bank["lit_ptr"][ids], bank["lit_ptr"][ids + 1]
    counts = ends - starts
    flat = np.concatenate([np.arange(s, e) for s, e in zip(starts, ends)])
    t = w.truth3(bank["lit_id"][flat])
    owner = np.repeat(np.arange(ids.shape[0]), counts)
    lo = np.full(ids.shape[0], TRUE, dtype=np.int8)
    hi = np.full(ids.shape[0], TRUE, dtype=np.int8)
    np.minimum.at(lo, owner, t)
    np.maximum.at(hi, owner, t)
    out[ids] = np.where(lo == FALSE, FALSE, np.where(hi == UNKNOWN, UNKNOWN, TRUE)).astype(np.int8)
    return out


# ---------------------------------------------------------------- decisions
def numerators(sums: np.ndarray, bank: dict) -> np.ndarray:
    return bank["a_q16"] * sums.astype(np.int64) * 256 + bank["b_q16"] * bank["class_bias"].astype(np.int64)


def probs(num: np.ndarray, temperature: float, allowed: np.ndarray | None) -> np.ndarray:
    # the same float64 operations, in the same order, as tmlm.decode.sampling.probs_from_numerators
    z = num.astype(np.float64) / (float(1 << 24) * float(temperature))
    if allowed is not None:
        z = np.where(allowed, z, -np.inf)
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


def selected(p: np.ndarray, u32: int) -> int:
    cdf = np.floor(np.cumsum(p) * float(TWO32)).astype(np.int64)
    cdf[-1] = TWO32
    return int(np.searchsorted(cdf, int(u32), side="right"))


def z3_argmax_ok(num: np.ndarray, claimed: int) -> bool:
    """Optional: z3 proves the claim "num[claimed] >= num[c] for every c, and > for c < claimed"
    is consistent (integer arithmetic), by checking that its negation is unsatisfiable."""
    import z3  # not a project import; optional
    s = z3.Solver()
    vals = [z3.IntVal(int(v)) for v in num]
    claim = z3.And(*[vals[claimed] >= vals[c] if c > claimed else vals[claimed] > vals[c] for c in range(len(vals)) if c != claimed])
    s.add(z3.Not(claim))
    return s.check() == z3.unsat


# ---------------------------------------------------------------- one token
def check_token(bank: dict, doc: dict, entry: dict, use_z3: bool = False) -> list[str]:
    """Empty list = pass; otherwise the reasons of failure."""
    errs: list[str] = []
    V = bank["vocab_size"]
    tokens = np.asarray(entry["window_tokens"], dtype=np.int64)
    hidden = np.asarray(entry["window_hidden"], dtype=np.int64)
    p = int(entry["position"])
    if tokens.shape != hidden.shape or not 0 <= p < tokens.shape[0]:
        return ["window: shape or position out of range"]
    if tokens.size and (tokens.min() < 0 or tokens.max() >= V):
        return ["window: token id out of range"]
    if not hidden[p]:
        errs.append("window: the target position is not hidden")
    w = Window(bank, tokens, hidden, p)
    listed = sorted(int(c["clause"]) for c in entry["fired"])
    if len(set(listed)) != len(listed):
        errs.append("fired: duplicate clause ids")
    # 2. listed clauses: alive, literals and votes as in the bank, satisfied
    sums = np.zeros(V, dtype=np.int64)
    for cl in entry["fired"]:
        j = int(cl["clause"])
        if not 0 <= j < bank["C"] or not bank["can_fire"][j]:
            errs.append(f"clause {j}: not an alive clause with an anchor")
            continue
        lits = clause_literals(bank, j)
        if [int(x) for x in lits] != [int(x) for x in cl["literals"]]:
            errs.append(f"clause {j}: literal list differs from the bank")
        c, wv = clause_votes(bank, j)
        if [[int(a), int(b)] for a, b in zip(c, wv)] != [[int(a), int(b)] for a, b in cl["votes"]]:
            errs.append(f"clause {j}: votes differ from the bank")
        if not np.all(w.truth(lits)):
            errs.append(f"clause {j}: not satisfied by the window")
        for a, b in zip(c, wv):
            sums[int(a)] += int(b)
    # 3. no unlisted clause fires
    fired = [int(j) for j in all_fired(bank, w)]
    if fired != listed:
        missing = sorted(set(fired) - set(listed))
        extra = sorted(set(listed) - set(fired))
        errs.append(f"fired set differs from the bank's: unlisted firing {missing[:5]}, listed not firing {extra[:5]}")
    # 4. sums
    rec = np.zeros(V, dtype=np.int64)
    for k, v in entry["sums"].items():
        rec[int(k)] = int(v)
    if not np.array_equal(rec, sums):
        errs.append("sums: not the sum of the listed votes")
    # 5. the decision
    claimed = int(entry["claimed"])
    if not 0 <= claimed < V:
        errs.append("claimed: out of range")
        return errs
    num = numerators(sums, bank)
    mode = entry["mode"]
    temperature = float.fromhex(entry["temperature_hex"])
    allowed = None
    if mode == "eliminate":
        el = entry["eliminate"]
        t3 = all_truth3(bank, w)
        lic, elim = [], []
        licensed = np.zeros(V, dtype=bool)
        eliminated = np.zeros(V, dtype=bool)
        for j in np.flatnonzero(t3 == TRUE):
            c, wv = clause_votes(bank, int(j))
            if np.any(wv > 0):
                lic.append(int(j))
                licensed[c[wv > 0]] = True
            if np.any(wv < 0):
                elim.append(int(j))
                eliminated[c[wv < 0]] = True
        survivors = (licensed if licensed.any() else np.ones(V, dtype=bool)) & ~eliminated
        conflict = not survivors.any()
        if conflict:
            survivors = np.ones(V, dtype=bool)
        if lic != [int(j) for j in el["licensing"]]:
            errs.append("eliminate: licensing clauses differ")
        if elim != [int(j) for j in el["eliminating"]]:
            errs.append("eliminate: eliminating clauses differ")
        if [int(c) for c in np.flatnonzero(survivors)] != [int(c) for c in el["survivors"]]:
            errs.append("eliminate: survivors differ")
        if bool(el["conflict"]) != conflict:
            errs.append("eliminate: conflict flag differs")
        if not survivors[claimed]:
            errs.append("eliminate: the claimed token is not a survivor")
        allowed = survivors
    if mode == "greedy":
        am = int(num.argmax())
        if am != claimed:
            errs.append(f"greedy: argmax is {am}, claimed {claimed}")
        if use_z3 and not z3_argmax_ok(num, claimed):
            errs.append("greedy: z3 refutes the argmax claim")
    elif mode in ("sample", "eliminate"):
        d = entry["draw"]
        if d is None:
            errs.append("sample: no draw recorded")
        else:
            u = rng32(int(d["seed"]), int(d["stream"]), int(d["a"]), int(d["b"]), int(d["c"]))
            if u != int(d["u32"]):
                errs.append("draw: the recorded 32-bit value is not rng32(seed, stream, a, b, c)")
            tok = selected(probs(num, temperature, allowed), u)
            if tok != claimed:
                errs.append(f"sample: the draw selects {tok}, claimed {claimed}")
    else:
        errs.append(f"unknown mode {mode!r}")
    return errs


# ---------------------------------------------------------------- binding to the generation run
def _int(value) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def doc_errors_items_1_to_5(bank: dict, doc: dict) -> list[str]:
    """The document checks of the checker as it ran in E11 (items 1 to 5): format, bank, state hash,
    calibration and vocabulary. Each one fails every token."""
    head: list[str] = []
    if doc.get("format") != "tmlm.receipt.trackB.v1":
        head.append("receipt format unknown")
    if doc.get("bank_sha256") != bank["_sha256"]:
        head.append("bank sha256 differs from the receipt")
    if doc.get("state_hash") != bank["state_hash"]:
        head.append("state_hash differs from the bank's")
    cal = doc.get("calibration", {})
    if int(cal.get("a_q16", -1)) != bank["a_q16"] or int(cal.get("b_q16", -1)) != bank["b_q16"]:
        head.append("calibration scalars differ from the bank's")
    if int(doc.get("vocab_size", -1)) != bank["vocab_size"]:
        head.append("vocab_size differs")
    return head


def doc_errors_binding(doc: dict, expected_seed: int | None = None, expected_temperature: float | None = None) -> list[str]:
    """The document checks of item 6: the generation record and, when given, the expected seed and temperature."""
    head: list[str] = []
    gen = doc.get("generation")
    if not isinstance(gen, dict) or gen.get("decoder") not in ("anyorder", "diffusion", "eliminate"):
        head.append("generation record missing or its decoder unknown")
        return head
    if _int(gen.get("seed")) is None:
        head.append("generation record: no integer seed")
    elif expected_seed is not None and gen["seed"] != int(expected_seed):
        head.append(f"generation seed {gen['seed']} is not the expected seed {int(expected_seed)}")
    if not isinstance(gen.get("temperature"), (int, float)) or isinstance(gen.get("temperature"), bool):
        head.append("generation record: no temperature")
    elif expected_temperature is not None and float(gen["temperature"]) != float(expected_temperature):
        head.append(f"generation temperature {gen['temperature']!r} is not the expected {float(expected_temperature)!r}")
    return head


def doc_errors(bank: dict, doc: dict, expected_seed: int | None = None,
               expected_temperature: float | None = None) -> list[str]:
    """Errors of the receipt document as a whole (items 1 to 5, then item 6); each one fails every token."""
    return doc_errors_items_1_to_5(bank, doc) + doc_errors_binding(doc, expected_seed, expected_temperature)


def rederivation_errors(bank: dict, doc: dict, i: int, use_z3: bool = False) -> list[str]:
    """Token i under items 1 to 5 only: exactly the checker as it ran in E11 (no binding to the run)."""
    head = doc_errors_items_1_to_5(bank, doc)
    return list(head) if head else check_token(bank, doc, doc["tokens"][i], use_z3)


def binding_only_errors(doc: dict, i: int, expected_seed: int | None = None,
                        expected_temperature: float | None = None) -> list[str]:
    """Token i under item 6 only: the generation record (and the expected values), then the binding."""
    head = doc_errors_binding(doc, expected_seed, expected_temperature)
    return list(head) if head else binding_errors(doc, i)


def binding_errors(doc: dict, i: int) -> list[str]:
    """Errors of token i against the run: mode, temperature, decoding step and draw counters."""
    gen, tokens = doc["generation"], doc["tokens"]
    entry = tokens[i]
    errs: list[str] = []
    decoder = gen["decoder"]
    modes = DECODER_MODES.get(decoder, ("greedy",) if gen.get("greedy") is True else ("sample",))
    if entry.get("mode") not in modes:
        errs.append(f"mode {entry.get('mode')!r} is not the {decoder} decoder's {modes[0]!r}")
    try:
        if float.fromhex(entry["temperature_hex"]) != float(gen["temperature"]):
            errs.append("temperature differs from the generation run's")
    except (KeyError, TypeError, ValueError):
        errs.append("temperature: not readable")
    step = _int(entry.get("step"))
    if step is None:
        errs.append("step: missing")
    elif decoder in ("anyorder", "eliminate"):
        if step != i:
            errs.append(f"step {step} is not the token's index {i} (one token per decoding step)")
    else:  # diffusion: steps 0, 1, 2, ... without a gap; one window per step
        prev = _int(tokens[i - 1].get("step")) if i > 0 else None
        if (i == 0 and step != 0) or (i > 0 and (prev is None or step - prev not in (0, 1))):
            errs.append(f"step {step} does not follow the step {prev} of the token before it")
        if any(k != i and tokens[k].get("step") == step
               and (tokens[k].get("window_tokens") != entry.get("window_tokens")
                    or tokens[k].get("window_hidden") != entry.get("window_hidden"))
               for k in range(len(tokens))):
            errs.append(f"window differs from the window of another token of step {step}")
        iterations = _int(gen.get("iterations"))
        if iterations is not None and (step >= iterations or (i == len(tokens) - 1 and step != iterations - 1)):
            errs.append(f"step {step} does not fit the run's {iterations} iterations")
    d = entry.get("draw")
    if entry.get("mode") in ("sample", "eliminate") and isinstance(d, dict):
        want = {"seed": gen.get("seed"), "stream": STREAM_SAMPLER, "a": step, "b": _int(entry.get("position")),
                "c": DRAW_TOKEN}
        wrong = [k for k, v in want.items() if _int(d.get(k)) is None or d.get(k) != v]
        if wrong:
            errs.append("draw: the counters " + ", ".join(wrong) + " are not those of this run, step and position "
                        "(seed of the run, stream SAMPLER, a = step, b = position, c = DRAW_TOKEN)")
    return errs


def token_errors(bank: dict, doc: dict, i: int, use_z3: bool = False, head: list[str] | None = None,
                 expected_seed: int | None = None, expected_temperature: float | None = None) -> list[str]:
    """Every error of token i: the document's, then the re-derivation (items 1 to 5), then the binding (6)."""
    head = doc_errors(bank, doc, expected_seed, expected_temperature) if head is None else head
    if head:
        return list(head)
    return check_token(bank, doc, doc["tokens"][i], use_z3) + binding_errors(doc, i)


def check_receipts(bank: dict, doc: dict, use_z3: bool = False, expected_seed: int | None = None,
                   expected_temperature: float | None = None) -> list[dict]:
    """Per-token results: {"index", "position", "claimed", "pass", "errors"}."""
    results = []
    head = doc_errors(bank, doc, expected_seed, expected_temperature)
    for i, entry in enumerate(doc.get("tokens", [])):
        errs = token_errors(bank, doc, i, use_z3, head)
        results.append({"index": i, "position": int(entry["position"]), "claimed": int(entry["claimed"]),
                        "pass": not errs, "errors": errs})
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--receipts", required=True)
    ap.add_argument("--seed", type=int, default=None, help="the seed the generation run must have (registered)")
    ap.add_argument("--temperature", type=float, default=None, help="the temperature the run must have")
    ap.add_argument("--z3", action="store_true", help="also verify greedy claims with z3 (optional package)")
    ap.add_argument("--json", default=None, help="write the per-token report here")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    bank = load_bank(args.bank)
    with open(args.receipts, encoding="utf-8") as f:
        loaded = json.load(f)
    docs = loaded["runs"] if isinstance(loaded, dict) and "runs" in loaded and "tokens" not in loaded else [loaded]
    results = []
    for run, doc in enumerate(docs):
        for r in check_receipts(bank, doc, args.z3, args.seed, args.temperature):
            results.append({"run": run, **r} if len(docs) > 1 else r)
    n_pass = sum(r["pass"] for r in results)
    if not args.quiet:
        for r in results:
            status = "PASS" if r["pass"] else "FAIL"
            where = f"run {r['run']:3d} " if "run" in r else ""
            print(f"{where}token {r['index']:4d} position {r['position']:5d} claimed {r['claimed']:6d} {status}" + ("" if r["pass"] else ": " + "; ".join(r["errors"])))
        print(f"{n_pass} of {len(results)} receipts pass")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"n_runs": len(docs), "n_tokens": len(results), "n_pass": n_pass, "results": results}, f, indent=1)
    return 0 if results and n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
