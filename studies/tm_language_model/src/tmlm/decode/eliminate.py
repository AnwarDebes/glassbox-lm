"""Elimination decoder (docs/03 "Certification spec" and "Six-challenge mapping", docs/04 T3.3,
docs/07 G1 "reasoning by elimination": positive clauses license tokens, negative clauses
eliminate; sample from the survivors weighted by votes).

Three-valued evaluation of the exported bank on a partial assignment (the window with its
hidden positions read as UNKNOWN rather than as the MASK symbol):
- an exact literal over a token or cluster symbol at a hidden position (other than the target)
  is UNKNOWN; a literal over MASK or EDGE is evaluated on the visibility as it stands (a hidden
  position IS a mask for the model that learned next to masks); a literal at a visible position
  is TRUE or FALSE as in docs/11 section 1.4;
- a positive bag literal is TRUE if a visible non-stop position of the bucket carries the
  symbol, else UNKNOWN if the bucket still holds a hidden position, else FALSE;
- a negated literal negates the three-valued value;
- a clause is FALSE if any literal is FALSE, TRUE if every literal is TRUE, else PENDING.

For the target position p: the TRUE clauses with positive votes LICENSE their voted classes,
the TRUE clauses with negative votes ELIMINATE theirs. Survivors are the licensed classes minus
the eliminated ones when anything is licensed, else every class minus the eliminated ones; an
empty survivor set is a CONFLICT and the decoder falls back to every class (reported, counted).
PySAT unit propagation runs over the CNF of these facts (one variable per class at p; the
at-least-one clause; one unit clause per eliminated class; one clause per licensing clause's
vote set): it reports forced classes and detects the conflict independently of the set
arithmetic (docs/04: "unit propagation over the exported bank with PySAT"). The token is then
drawn from the calibrated distribution restricted to the survivors, counter-based RNG.
The receipt records the licensing and eliminating clause ids, the survivors and the draw; the
standalone checker re-derives the three-valued sets from the bank and the window.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..engine.config import ANCHOR_NONE, N_EXACT_SEG, bag_of_seg, bag_range
from .anyorder import prepare_window
from .bank import Bank, WindowEval
from .conftm import logit_numerators
from .sampling import DRAW_TOKEN, draw, sample_token
from .trace import Generation, StepRecord

TRUE, FALSE, UNKNOWN = 1, 0, 2


@dataclass
class Elimination:
    position: int
    licensing: np.ndarray  # u32 ascending clause ids
    eliminating: np.ndarray  # u32 ascending
    pending: int
    licensed: np.ndarray  # bool [V]
    eliminated: np.ndarray  # bool [V]
    survivors: np.ndarray  # bool [V]
    conflict: bool
    forced: list[int]  # classes forced true by unit propagation
    propagation_status: str  # "ok" | "conflict" | "unavailable"


def literal_truth3(we: WindowEval, p: int, lits: np.ndarray) -> np.ndarray:
    """int8 [n] in {TRUE, FALSE, UNKNOWN} for target p on the window of `we` (vectorised)."""
    lits = np.asarray(lits, dtype=np.uint32)
    seg = (lits >> 17).astype(np.int64)
    sym = ((lits >> 1) & 0xFFFF).astype(np.int64)
    neg = (lits & 1).astype(bool)
    two = we.literal_truth(p, lits & ~np.uint32(1))  # truth of the POSITIVE forms in the two-valued model
    out = np.where(two, TRUE, FALSE).astype(np.int8)
    smask, sedge = we.smask, we.sedge
    exact = seg < N_EXACT_SEG
    if exact.any():
        e = np.flatnonzero(exact)
        q = p + we.offsets[seg[e]]
        inside = (q >= 0) & (q < we.L) & (q != p)
        hid = np.zeros(e.shape[0], dtype=bool)
        hid[inside] = we.hidden[q[inside]] != 0
        special = (sym[e] == smask) | (sym[e] == sedge)
        unknown = hid & ~special & we.enabled[seg[e]]
        out[e[unknown]] = UNKNOWN
    bag = ~exact
    if bag.any():
        b = np.flatnonzero(bag)
        for s in np.unique(seg[b]):
            s = int(s)
            if s >= we.enabled.shape[0] or not we.enabled[s]:
                continue
            k, side = bag_of_seg(s)
            lo, hi = bag_range(k)
            first, last = (p - hi, p - lo) if side == 0 else (p + lo, p + hi)
            first, last = max(first, 0), min(last, we.L - 1)
            any_hidden = False
            if first <= last:
                q = np.arange(first, last + 1)
                any_hidden = bool(np.any(we.hidden[q[q != p]]))
            if any_hidden:
                sel = b[seg[b] == s]
                out[sel[out[sel] != TRUE]] = UNKNOWN
    flip = neg & (out != UNKNOWN)
    out[flip] = 1 - out[flip]
    return out


def clause_truth3(bank: Bank, we: WindowEval, p: int) -> np.ndarray:
    """int8 [C]: three-valued truth of every clause for target p (FALSE for dead clauses)."""
    C = bank.n_clauses
    out = np.full(C, FALSE, dtype=np.int8)
    ids = np.flatnonzero((bank.alive != 0) & (bank.anchor != ANCHOR_NONE))
    if ids.shape[0] == 0:
        return out
    starts, ends = bank.lit_ptr[ids], bank.lit_ptr[ids + 1]
    counts = ends - starts
    flat = np.concatenate([np.arange(s, e) for s, e in zip(starts, ends)])
    t = literal_truth3(we, p, bank.lit_id[flat])
    owner = np.repeat(np.arange(ids.shape[0]), counts)
    lo = np.full(ids.shape[0], TRUE, dtype=np.int8)
    hi = np.full(ids.shape[0], TRUE, dtype=np.int8)
    np.minimum.at(lo, owner, t)  # 0 if any FALSE
    np.maximum.at(hi, owner, t)  # 2 if any UNKNOWN
    val = np.where(lo == FALSE, FALSE, np.where(hi == UNKNOWN, UNKNOWN, TRUE)).astype(np.int8)
    out[ids] = val
    return out


def eliminate(bank: Bank, we: WindowEval, p: int, propagate: bool = True) -> Elimination:
    truth = clause_truth3(bank, we, p)
    V = bank.vocab
    licensed = np.zeros(V, dtype=bool)
    eliminated = np.zeros(V, dtype=bool)
    lic, elim = [], []
    for j in np.flatnonzero(truth == TRUE):
        c, w = bank.votes(int(j))
        if np.any(w > 0):
            lic.append(int(j))
            licensed[c[w > 0]] = True
        if np.any(w < 0):
            elim.append(int(j))
            eliminated[c[w < 0]] = True
    survivors = (licensed if licensed.any() else np.ones(V, dtype=bool)) & ~eliminated
    conflict = not survivors.any()
    if conflict:
        survivors = np.ones(V, dtype=bool)
    forced, status = [], "unavailable"
    if propagate:
        forced, status = unit_propagate(bank, lic, elim, V)
    return Elimination(position=int(p), licensing=np.asarray(lic, dtype=np.uint32), eliminating=np.asarray(elim, dtype=np.uint32),
                       pending=int((truth == UNKNOWN).sum()), licensed=licensed, eliminated=eliminated, survivors=survivors,
                       conflict=conflict, forced=forced, propagation_status=status)


def unit_propagate(bank: Bank, licensing: list[int], eliminating: list[int], V: int) -> tuple[list[int], str]:
    """PySAT unit propagation over the per-token CNF. Variables 1..V = "class c at p"."""
    try:
        from pysat.solvers import Solver
    except ImportError:
        return [], "unavailable"
    cnf = [list(range(1, V + 1))]
    for j in eliminating:
        c, w = bank.votes(int(j))
        cnf.extend([[-(int(cc) + 1)] for cc in c[w < 0]])
    for j in licensing:
        c, w = bank.votes(int(j))
        cnf.append([int(cc) + 1 for cc in c[w > 0]])
    with Solver(name="m22", bootstrap_with=cnf) as s:
        ok, assigned = s.propagate()
        if not ok:
            return [], "conflict"
        forced = sorted(int(v) - 1 for v in assigned if v > 0)
        return forced, "ok"


def elimination_generate(scorer, prompt: np.ndarray, n_new: int, seed: int, temperature: float = 1.0,
                         propagate: bool = True, record: bool = True) -> Generation:
    """Left-to-right elimination decoding: survivors from the three-valued bank, the token drawn
    from the calibrated engine distribution restricted to them."""
    model = scorer.model
    bank = model.bank()
    t0 = time.time()
    calls0 = getattr(scorer, "n_engine_calls", 0)
    gen_pos = prepare_window(scorer, prompt, n_new)
    we = WindowEval(bank, scorer.tokens.copy(), scorer.hidden.copy())
    out = Generation(prompt=np.asarray(prompt, dtype=np.uint16), tokens=np.zeros(scorer.L, dtype=np.uint16), generated_positions=gen_pos)
    cal, bias = model.calibration, model.class_bias
    conflicts = 0
    for step, p in enumerate(gen_pos.tolist()):
        el = eliminate(bank, we, p, propagate)
        conflicts += int(el.conflict)
        row = scorer.scores(np.array([p]))[0]
        num = logit_numerators(row[None, :], bias, cal.a_q16, cal.b_q16)[0]
        d = draw(seed, step, p, DRAW_TOKEN)
        claimed = sample_token(num, temperature, d, allowed=el.survivors)
        if record:
            out.records.append(StepRecord(step=step, position=p, claimed=claimed, mode="eliminate",
                                          tokens=scorer.tokens.copy(), hidden=scorer.hidden.copy(),
                                          fired=np.asarray(scorer.fired(p), dtype=np.uint32), sums=row.copy(),
                                          temperature=float(temperature), draw=d, licensing=el.licensing,
                                          eliminating=el.eliminating, pending=el.pending, survivors=el.survivors.copy(),
                                          conflict=el.conflict))
        scorer.set_token(p, claimed)
        we.set_token(p, claimed)
    out.tokens = scorer.tokens.copy()
    out.seconds = time.time() - t0
    out.engine_calls = getattr(scorer, "n_engine_calls", 0) - calls0
    out.info = {"decoder": "eliminate", "temperature": float(temperature), "seed": int(seed), "conflicts": conflicts, "propagate": bool(propagate)}
    return out
