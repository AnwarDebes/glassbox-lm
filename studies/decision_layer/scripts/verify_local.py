#!/usr/bin/env python
"""e71 (plan logs/plan_e71_local.md): local guideline proofs around real dev states, in word edits.

For a real dev state (its memory as logged) and a group of buttons that the rules forbid in that state: the fewest
word edits to the customer's latest utterance that make the model choose one of them. An edit deletes one of the
utterance's words, or inserts a candidate word at its end. One MILP per (state, group) minimises the edit count, which
gives every k at once.

    verify_local.py --check model --n 200        # pre-check of the text model (exact bits, with and without edits)
    verify_local.py --check run --model tm --seed 1

The text model of view B (mem, words, xcust, xrecent), for state r with latest customer utterance u:
- xcust and xrecent (lookup + OR of per-token codes): the bits of the fixed text, OR the codes of the kept words of u,
  OR the codes of the inserted words (each word's codes are those of its table tokens). Exact.
- words, unigrams: present in the fixed text, or a kept or inserted word. Exact.
- words, bigrams: those inside the fixed text stay on. Every other lexicon bigram (a, b) may be on only if a is a word of
  the region (the kept and inserted words, or the fixed word just before u) and b is too (or the fixed word just after u),
  and it equals the real state's bit when no edit is made. A superset of the real outcomes (the insertion order is free).
- u keeps at least one word (an empty utterance would read as "(nothing yet)").
- If u is not in the last 3 turns, only xcust changes.
A certificate (no edit set of size < k makes a forbidden button win) is sound for every real edit; a counterexample is
replayed through the real pipeline (the edited text, then view.transform, then the model) and is called realizable
only if the violation reproduces. The TM encoding, winner rows and solver options are e70's (scripts/verify_policy.py).
Results: one JSON line per (state, group) in results/explore/verify_e71.jsonl. Dev only. Added 2026-10-19.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, milp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(1, str(ROOT / "scripts"))

from verify_policy import B_, HIGHS_OPTS, N_MEM, SOLVER, SRC, domain_rows, load_tm  # noqa: E402

OUT = ROOT / "results" / "explore" / "verify_e71.jsonl"
WORDRE = re.compile(r"(?u)\b\w+\b")
CAND_MIN_UTT = 5  # candidate insertions: words in at least 5 train customer utterances (plan, fixed)


# ---------- the view and the per-word codes ----------

class TextModel:
    def __init__(self):
        from dl.abcd import ABCDView, load_examples  # noqa: E402
        from dl.static_text import table  # noqa: E402

        self.tr, self.blist = load_examples("train")
        self.dv, _ = load_examples("dev")
        E = np.load(ROOT / "data" / "emb" / "abcd__potion-32m.npz")
        self.em = {s: {f: E[f"{s}__{f}"] for f in ("cust", "recent", "full")} for s in ("train", "dev")}
        self.view = ABCDView(parts=("mem", "words", "xcust", "xrecent"), lex_k=1000, seed=0).fit(self.tr, self.em["train"], self.blist)
        self.names = self.view.names
        assert self.names == json.loads((SRC / "bit_names.json").read_text())
        self.tb = table()
        self.vocab = self.view.lex.vec.vocabulary_  # n-gram -> column
        self.o_words = N_MEM
        self.o_cust = N_MEM + len(self.vocab)
        self.o_recent = self.o_cust + self.view.xcust.n_bits
        self.F = self.o_recent + self.view.xrecent.n_bits
        assert self.F == len(self.names)
        self.bits = {"xcust": np.unpackbits(self.view.xcust.packed, axis=1)[:, : self.view.xcust.n_bits].astype(bool),
                     "xrecent": np.unpackbits(self.view.xrecent.packed, axis=1)[:, : self.view.xrecent.n_bits].astype(bool)}
        last = {}
        for r in self.tr:
            last[r["convo_id"]] = r["cust"]
        cnt = Counter(w for c in last.values() for u in c.split(" | ") for w in set(WORDRE.findall(u.lower())))
        self.cand = sorted(w for w, n in cnt.items() if n >= CAND_MIN_UTT)
        self._codes = {}
        self._tcode = {}
        self.bigrams_with: dict[str, list[str]] = {}
        for g in self.vocab:
            if " " in g:
                for w in set(g.split(" ")):
                    self.bigrams_with.setdefault(w, []).append(g)

    def codes(self, word: str) -> dict[str, np.ndarray]:
        """The bits one word sets in each word-table field (indices within the field)."""
        if word not in self._codes:
            toks = self.tb.ids([word])[0]
            self._codes[word] = {f: np.flatnonzero(self.bits[f][toks].any(0)) for f in ("xcust", "xrecent")}
        return self._codes[word]

    def tokens_code(self, text: str, field: str) -> np.ndarray:
        key = (text, field)
        if key not in self._tcode:
            if not text.strip():
                self._tcode[key] = np.zeros(self.bits[field].shape[1], bool)
            else:
                toks = self.tb.ids([text])[0]
                self._tcode[key] = self.bits[field][toks].any(0) if toks else np.zeros(self.bits[field].shape[1], bool)
        return self._tcode[key]

    def split(self, r: dict) -> dict:
        """The fixed text and the editable utterance u of state r."""
        parts = r["cust"].split(" | ")
        u = parts[-1]
        residue = WORDRE.sub(" ", u)
        cust_fixed = (" | ".join(parts[:-1]) + " | " if len(parts) > 1 else "") + residue
        rec = r["recent"]
        key = "customer: " + u
        i = rec.rfind(key)
        in_recent = i >= 0 and u != ""
        s = {"u": u, "occ": WORDRE.findall(u.lower()), "cust_fixed": cust_fixed, "in_recent": in_recent}
        if in_recent:
            before, after = rec[: i + len("customer: ")], rec[i + len(key):]
            s.update(before=before, after=after, recent_fixed=before + residue + after)
            tb_, ta_ = WORDRE.findall(before.lower()), WORDRE.findall(after.lower())
            s["fixed_uni"] = set(tb_) | set(ta_)
            s["fixed_bi"] = {f"{a} {b}" for seq in (tb_, ta_) for a, b in zip(seq, seq[1:])}
            s["jb"], s["ja"] = (tb_[-1] if tb_ else None), (ta_[0] if ta_ else None)
        return s

    def edited_row(self, r: dict, keep: list[bool], ins: list[str]) -> dict:
        """The real edited conversation state: u's words deleted where keep is False, the inserted words appended to u."""
        s = self.split(r)
        u = s["u"]
        out, k = [], 0
        pos = 0
        for m in WORDRE.finditer(u):
            out.append(u[pos : m.start()])
            out.append(m.group(0) if keep[k] else "")
            pos, k = m.end(), k + 1
        out.append(u[pos:])
        u2 = "".join(out)
        if ins:
            u2 = (u2.rstrip() + " " + " ".join(ins)).strip() if u2.strip() else " ".join(ins)
        parts = r["cust"].split(" | ")
        r2 = dict(r, cust=" | ".join(parts[:-1] + [u2]))
        if s["in_recent"]:
            r2["recent"] = s["before"] + u2 + s["after"]
        return r2

    def model_bits(self, r: dict, keep: list[bool], ins: list[str], order_known: bool = True, x0: np.ndarray | None = None) -> np.ndarray:
        """The text model's bits for given edits (bigrams: the real adjacency when the insertion order is known)."""
        s = self.split(r)
        bits = np.zeros(self.F, np.uint8)
        if not s["in_recent"]:  # only xcust changes: the words and xrecent bits are the real state's
            if x0 is None:
                x0 = self.view.transform([r], {f: v[:1] for f, v in self.em["dev"].items()})[0]
            bits[self.o_words : self.o_cust] = x0[self.o_words : self.o_cust]
            bits[self.o_recent : self.F] = x0[self.o_recent : self.F]
        kept = [w for w, k in zip(s["occ"], keep) if k]
        words = kept + list(ins)
        cust = self.tokens_code(s["cust_fixed"], "xcust").copy()
        for w in words:
            cust[self.codes(w)["xcust"]] = True
        bits[self.o_cust : self.o_recent] = cust
        if s["in_recent"]:
            rec = self.tokens_code(s["recent_fixed"], "xrecent").copy()
            for w in words:
                rec[self.codes(w)["xrecent"]] = True
            bits[self.o_recent : self.F] = rec
            seq = ([s["jb"]] if s["jb"] else []) + words + ([s["ja"]] if s["ja"] else [])
            uni = s["fixed_uni"] | set(words)
            bi = s["fixed_bi"] | {f"{a} {b}" for a, b in zip(seq, seq[1:])}
            for g, c in self.vocab.items():
                if g in uni or g in bi:
                    bits[self.o_words + c] = 1
        return bits


# ---------- the MILP: fewest edits to make some button of a group win ----------

def used_bits(tm: dict) -> np.ndarray:
    """[F] bool: the bits that some clause of the model reads (as a literal or its negation)."""
    if "used" not in tm:
        F = tm["F"]
        U = np.zeros(F, bool)
        for lits in tm["clauses"]:
            for l in lits:
                U[l if l < F else l - F] = True
        tm["used"] = U
    return tm["used"]


def solve_group(tmod: TextModel, tm: dict, r: dict, x0: np.ndarray, group: list[int], against: dict[int, list[int]],
                time_limit: float, threads: int, fix_edits: tuple | None = None, real_point: tuple | None = None,
                budget: int | None = None) -> dict:
    """real_point = (keep flags, inserted words, the real edited bit vector, the winning button): no solve; instead the
    MILP's rows are evaluated at the real outcome (a pre-check that the encoding contains it)."""
    s = tmod.split(r)
    F, W = tm["F"], tm["W"]
    U = used_bits(tm)  # bits that some clause reads (e71 note 1: a bit no clause reads cannot change a vote)
    occ = s["occ"]
    bases = {"xcust": tmod.tokens_code(s["cust_fixed"], "xcust")}
    if s["in_recent"]:
        bases["xrecent"] = tmod.tokens_code(s["recent_fixed"], "xrecent")

    def relevant(w: str) -> bool:
        """Can inserting w change a bit that some clause reads? (exact pruning, e71 note 1)"""
        for field, off in (("xcust", tmod.o_cust), ("xrecent", tmod.o_recent)):
            if field in bases and any(U[off + int(j)] and not bases[field][int(j)] for j in tmod.codes(w)[field]):
                return True
        if s["in_recent"]:
            if w in tmod.vocab and " " not in w and U[tmod.o_words + tmod.vocab[w]] and w not in s["fixed_uni"]:
                return True
            if any(U[tmod.o_words + tmod.vocab[g]] and g not in s["fixed_bi"] for g in tmod.bigrams_with.get(w, [])):
                return True
        return False

    cand = [w for w in tmod.cand if relevant(w)]
    has_filler = len(cand) < len(tmod.cand)  # an irrelevant word can still be inserted, at one edit, with no effect
    n_o, n_c = len(occ), len(cand)
    # columns: keep (n_o) | ins (n_c) | free text bits | clauses | y (group)
    col_keep = {o: o for o in range(n_o)}
    col_ins = {w: n_o + i for i, w in enumerate(cand)}
    col_E = n_o + n_c  # the edit count, E = n_o - sum keep + sum ins (one defining row; added 2026-10-19, e71 note 1)
    col_fill = n_o + n_c + 1  # the filler insertion (an irrelevant word)
    ncol = n_o + n_c + 2
    fixed = {i: int(x0[i]) for i in range(N_MEM)}  # memory as logged
    rows_r, rows_c, rows_v, lo, hi = [], [], [], [], []
    nrow = 0
    rtype, cur = [], ["?"]

    def add(entries, lo_, hi_):
        nonlocal nrow
        rtype.append(cur[0])
        for c_, v_ in entries:
            rows_r.append(nrow)
            rows_c.append(c_)
            rows_v.append(v_)
        lo.append(lo_)
        hi.append(hi_)
        nrow += 1

    free_col = {}  # bit index -> column
    bin_bits = []  # bigram bits are binary
    cur[0] = "edit_count"
    add([(col_E, 1)] + [(col_keep[o], 1) for o in range(n_o)] + [(col_ins[w], -1) for w in cand] + [(col_fill, -1)], n_o, n_o)
    # word-table fields
    for field, off, active in (("xcust", tmod.o_cust, True), ("xrecent", tmod.o_recent, s["in_recent"])):
        nb = tmod.bits[field].shape[1]
        if not active:
            for j in range(nb):
                fixed[off + j] = int(x0[off + j])
            continue
        base = bases[field]
        contrib: dict[int, list[int]] = {}
        for o, w in enumerate(occ):
            for j in tmod.codes(w)[field]:
                contrib.setdefault(int(j), []).append(col_keep[o])
        for w in cand:
            for j in tmod.codes(w)[field]:
                contrib.setdefault(int(j), []).append(col_ins[w])
        occ_set = {col_keep[o] for o in range(n_o)}
        for j in range(nb):
            if not U[off + j]:
                continue  # no clause reads this bit
            if base[j]:
                fixed[off + j] = 1
            elif j not in contrib:
                fixed[off + j] = 0
            else:
                c = ncol
                free_col[off + j] = c
                ncol += 1
                cur[0] = f"{field}_ge"
                for cc in contrib[j]:
                    if cc in occ_set:
                        add([(c, 1), (cc, -1)], 0, np.inf)  # x >= keep_o (one row per kept word and bit)
                cur[0] = f"{field}_le"
                add([(c, 1)] + [(cc, -1) for cc in contrib[j]], -np.inf, 0)  # x <= sum
        # an inserted word turns on all of its free bits: sum of its free bits >= (their number) x ins_w (one row per word)
        cur[0] = f"{field}_ins"
        for w in cand:
            cols = [free_col[off + int(j)] for j in tmod.codes(w)[field] if (off + int(j)) in free_col]
            if cols:
                add([(cc, 1) for cc in cols] + [(col_ins[w], -len(cols))], 0, np.inf)
    # words field
    if not s["in_recent"]:
        for c in tmod.vocab.values():
            fixed[tmod.o_words + c] = int(x0[tmod.o_words + c])
    else:
        occ_cols: dict[str, list[int]] = {}
        for o, w in enumerate(occ):
            occ_cols.setdefault(w, []).append(col_keep[o])
        region = set(occ) | set(cand)
        for g, c in tmod.vocab.items():
            i = tmod.o_words + c
            if not U[i]:
                continue  # no clause reads this bit
            if " " not in g:
                if g in s["fixed_uni"]:
                    fixed[i] = 1
                    continue
                cols = occ_cols.get(g, []) + ([col_ins[g]] if g in col_ins else [])
                if not cols:
                    fixed[i] = 0
                    continue
                cc = ncol
                free_col[i] = cc
                ncol += 1
                cur[0] = "uni_ge"
                for k in cols:
                    add([(cc, 1), (k, -1)], 0, np.inf)
                cur[0] = "uni_le"
                add([(cc, 1)] + [(k, -1) for k in cols], -np.inf, 0)
            else:
                a, b = g.split(" ")
                if g in s["fixed_bi"]:
                    fixed[i] = 1
                    continue
                left_ok = (a == s["jb"]) or a in region
                right_ok = (b == s["ja"]) or b in region
                if not (left_ok and right_ok):
                    fixed[i] = int(x0[i])  # cannot change (x0 then has it off, or on from the fixed text)
                    continue
                cc = ncol
                free_col[i] = cc
                bin_bits.append(cc)
                ncol += 1
                for side, w, jw in ((0, a, s["jb"]), (1, b, s["ja"])):
                    if w == jw:
                        continue  # the fixed neighbour is always there
                    cols = occ_cols.get(w, []) + ([col_ins[w]] if w in col_ins else [])
                    cur[0] = "bi_avail"
                    add([(cc, 1)] + [(k, -1) for k in cols], -np.inf, 0)  # x_ab <= avail(w)
                # equal to the real bit when no edit is made: |x - x0| <= E
                x0b = int(x0[i])
                cur[0] = "bi_noedit"
                add([(cc, 1), (col_E, -1)], -np.inf, x0b)  # x - x0 <= E
                add([(cc, -1), (col_E, -1)], -np.inf, -x0b)  # x0 - x <= E
    # u keeps at least one word: sum keep + sum ins >= 1
    cur[0] = "nonempty"
    add([(col_keep[o], 1) for o in range(n_o)] + [(col_ins[w], 1) for w in cand] + [(col_fill, 1)], 1, np.inf)
    assert all((i in fixed) or (i in free_col) for i in np.flatnonzero(U)), "a used bit is neither fixed nor free"
    # clauses, with the fixed bits evaluated away
    const = np.zeros(W.shape[1], np.int64)
    active = []
    for j, lits in enumerate(tm["clauses"]):
        keep_l, dead = [], False
        for l in lits:
            i, pos = (l, True) if l < F else (l - F, False)
            if i in fixed:
                if (fixed[i] == 1) != pos:
                    dead = True
                    break
                continue
            keep_l.append((free_col[i], pos))
        if dead:
            continue
        if not keep_l:
            const += W[j]
            continue
        active.append((j, keep_l))
    c0 = ncol
    ncol += len(active)
    cur[0] = "clause"
    for a_, (j, keep_l) in enumerate(active):
        cj = c0 + a_
        for xc, pos in keep_l:
            add([(cj, 1), (xc, -1)], -np.inf, 0) if pos else add([(cj, 1), (xc, 1)], -np.inf, 1)
        n_neg = sum(not p for _, p in keep_l)
        add([(cj, 1)] + [(xc, -1 if p else 1) for xc, p in keep_l], 1 - len(keep_l) + n_neg, np.inf)
    # some button of the group wins (big-M per pair, exact bounds from the active clauses)
    y0 = ncol
    ncol += len(group)
    cur[0] = "winner"
    add([(y0 + g, 1) for g in range(len(group))], 1, np.inf)
    jidx = np.array([j for j, _ in active], np.int64)
    for g, B in enumerate(group):
        for k in against[B]:
            if k == B:
                continue
            d = W[jidx, B] - W[jidx, k] if len(jidx) else np.zeros(0, np.int64)
            theta = 1 if k < B else 0
            M = int(np.clip(-d, 0, None).sum()) + int(max(0, const[k] - const[B])) + theta
            add([(c0 + a_, int(d[a_])) for a_ in range(len(active)) if d[a_] != 0] + [(y0 + g, -M)], theta - M - int(const[B] - const[k]), np.inf)
    A = sp.csr_matrix((rows_v, (rows_r, rows_c)), shape=(nrow, ncol))
    if real_point is not None:
        keep_r, ins_r, xr, win = real_point
        v = np.zeros(ncol)
        for o, k in enumerate(keep_r):
            v[col_keep[o]] = float(k)
        for w in ins_r:
            if w in col_ins:
                v[col_ins[w]] = 1.0
        v[col_fill] = float(sum(w not in col_ins for w in ins_r))
        v[col_E] = float(n_o - sum(keep_r) + len(ins_r))
        for i, c in free_col.items():
            v[c] = float(xr[i])
        fixed_ok = all(int(xr[i]) == val for i, val in fixed.items())
        lit = lambda xc, pos: v[xc] if pos else 1.0 - v[xc]  # noqa: E731
        for a_, (j, keep_l) in enumerate(active):
            v[c0 + a_] = float(all(lit(xc, p) == 1.0 for xc, p in keep_l))
        if win in group:
            v[y0 + group.index(win)] = 1.0
        Av = A @ v
        badm = (Av < np.array(lo) - 1e-9) | (Av > np.array(hi) + 1e-9)
        types = Counter(rtype[q] for q in np.flatnonzero(badm))
        return {"fixed_bits_match": bool(fixed_ok), "rows_broken": int(badm.sum()), "n_rows": nrow, "broken_types": dict(types)}
    lbv, ubv = np.zeros(ncol), np.ones(ncol)
    ubv[col_E] = n_o + n_c + 1
    if budget is not None:  # feasibility with at most `budget` edits (no objective)
        ubv[col_E] = budget
    ubv[col_fill] = len(tmod.cand) - n_c  # how many irrelevant words could be inserted (0 if none)
    if fix_edits is not None:  # pre-check: the edits fixed (keep flags, inserted words)
        keep_f, ins_f = fix_edits
        for o, k in enumerate(keep_f):
            lbv[col_keep[o]] = ubv[col_keep[o]] = float(k)
        for w in cand:
            v = float(w in ins_f)
            lbv[col_ins[w]] = ubv[col_ins[w]] = v
        n_irr = sum(w not in col_ins for w in ins_f)
        lbv[col_fill] = ubv[col_fill] = float(n_irr)
        lbv[col_E] = ubv[col_E] = float(n_o - sum(keep_f) + len(ins_f))
    integ = np.zeros(ncol)
    integ[: n_o + n_c] = 1
    integ[col_fill] = 1
    integ[bin_bits] = 1
    integ[y0:] = 1
    cost = np.zeros(ncol)
    cost[col_E] = 1  # the edit count
    t0 = time.time()
    res = milp(c=cost, constraints=LinearConstraint(A, np.array(lo, float), np.array(hi, float)), integrality=integ,
               bounds=Bounds(lbv, ubv),
               options={"time_limit": time_limit, "disp": False, "presolve": True, "threads": threads, **HIGHS_OPTS})
    out = {"seconds": round(time.time() - t0, 2), "highs_status": int(res.status), "message": str(res.message), "n_occ": n_o,
           "n_cand": n_c, "n_free_bits": len(free_col), "n_active_clauses": len(active), "n_rows": nrow, "nnz": int(A.nnz),
           "nodes": int(getattr(res, "mip_node_count", -1) or -1), "in_recent": s["in_recent"],
           "solver": {**SOLVER, "time_limit": time_limit, "threads": threads, **HIGHS_OPTS}}
    dual = getattr(res, "mip_dual_bound", None)
    out["dual_bound_edits"] = None if dual is None or not np.isfinite(dual) else float(dual)
    out.update(n_cand_relevant=n_c, filler=bool(has_filler), n_used_bits=int(U[N_MEM:].sum()))
    if res.x is not None and res.status in (0, 1):
        keep = [bool(round(res.x[o])) for o in range(n_o)]
        ins = [w for w in cand if round(res.x[col_ins[w]])]
        nf = int(round(res.x[col_fill]))
        if nf:
            ins += [w for w in tmod.cand if w not in col_ins][:nf]  # the filler: irrelevant words
        out.update(edits=int(n_o - sum(keep) + len(ins)), deleted=[w for w, k in zip(occ, keep) if not k], inserted=ins, keep=keep,
                   chosen=[tmod.blist[B] for g, B in enumerate(group) if round(res.x[y0 + g])])
    if res.status == 0:
        out["status"] = "optimal"
    elif res.status == 2 or "infeasible" in str(res.message).lower():
        out["status"] = "infeasible"
    elif res.status == 1:
        out["status"] = "time limit"
    else:
        out["status"] = f"other ({res.status})"
    return out


def solve_bits(tm: dict, rows_dom, x0: np.ndarray, group: list[int], against: dict[int, list[int]], time_limit: float,
               threads: int) -> dict:
    """Secondary measure: the fewest text-bit flips (memory fixed as logged, Level-B domain) that make some button of the
    group win. e70's TM encoding with a Hamming objective."""
    F, W = tm["F"], tm["W"]
    fixed = {i: int(x0[i]) for i in range(N_MEM)}
    free = [i for i in range(N_MEM, F)]
    col = {i: k for k, i in enumerate(free)}
    rr, cc, vv, lo, hi = [], [], [], [], []
    nrow = 0

    def add(entries, lo_, hi_):
        nonlocal nrow
        for c_, v_ in entries:
            rr.append(nrow); cc.append(c_); vv.append(v_)
        lo.append(lo_); hi.append(hi_)
        nrow += 1

    const = np.zeros(W.shape[1], np.int64)
    active = []
    for j, lits in enumerate(tm["clauses"]):
        keep_l, dead = [], False
        for l in lits:
            i, pos = (l, True) if l < F else (l - F, False)
            if i in fixed:
                if (fixed[i] == 1) != pos:
                    dead = True
                    break
                continue
            keep_l.append((col[i], pos))
        if dead:
            continue
        if not keep_l:
            const += W[j]
            continue
        active.append((j, keep_l))
    nx = len(free)
    c0 = nx
    for a_, (j, keep_l) in enumerate(active):
        cj = c0 + a_
        for xc, pos in keep_l:
            add([(cj, 1), (xc, -1)], -np.inf, 0) if pos else add([(cj, 1), (xc, 1)], -np.inf, 1)
        n_neg = sum(not p for _, p in keep_l)
        add([(cj, 1)] + [(xc, -1 if p else 1) for xc, p in keep_l], 1 - len(keep_l) + n_neg, np.inf)
    y0 = c0 + len(active)
    ncol = y0 + len(group)
    add([(y0 + g, 1) for g in range(len(group))], 1, np.inf)
    jidx = np.array([j for j, _ in active], np.int64)
    for g, B in enumerate(group):
        for k in against[B]:
            if k == B:
                continue
            d = W[jidx, B] - W[jidx, k]
            theta = 1 if k < B else 0
            M = int(np.clip(-d, 0, None).sum()) + int(max(0, const[k] - const[B])) + theta
            add([(c0 + a_, int(d[a_])) for a_ in range(len(active)) if d[a_] != 0] + [(y0 + g, -M)], theta - M - int(const[B] - const[k]), np.inf)
    for coef, lo_, hi_ in rows_dom:
        ent, shift = [], 0
        for i, cf in coef.items():
            if i in col:
                ent.append((col[i], cf))
            else:
                shift += cf * fixed[i]
        if ent:
            add(ent, lo_ - shift, hi_ - shift)
    A = sp.csr_matrix((vv, (rr, cc)), shape=(nrow, ncol))
    integ = np.zeros(ncol)
    integ[:nx] = 1
    integ[y0:] = 1
    cost = np.zeros(ncol)
    x0t = x0[N_MEM:F].astype(float)
    cost[:nx] = 1 - 2 * x0t  # Hamming: sum x over x0 = 0, plus sum (1 - x) over x0 = 1 (constant added back)
    t0 = time.time()
    res = milp(c=cost, constraints=LinearConstraint(A, np.array(lo, float), np.array(hi, float)), integrality=integ,
               bounds=Bounds(np.zeros(ncol), np.ones(ncol)),
               options={"time_limit": time_limit, "disp": False, "presolve": True, "threads": threads, **HIGHS_OPTS})
    out = {"seconds": round(time.time() - t0, 2), "highs_status": int(res.status), "message": str(res.message), "n_free_bits": nx,
           "n_active_clauses": len(active), "n_rows": nrow, "nodes": int(getattr(res, "mip_node_count", -1) or -1),
           "solver": {**SOLVER, "time_limit": time_limit, "threads": threads, **HIGHS_OPTS}}
    dual = getattr(res, "mip_dual_bound", None)
    out["dual_bound_flips"] = None if dual is None or not np.isfinite(dual) else float(dual + x0t.sum())
    if res.x is not None and res.status in (0, 1):
        x = x0.copy()
        x[N_MEM:F] = np.round(res.x[:nx]).astype(np.uint8)
        out["flips"] = int((x[N_MEM:F] != x0[N_MEM:F]).sum())
        out["x"] = x
    out["status"] = {0: "optimal", 1: "time limit"}.get(int(res.status), "infeasible" if (res.status == 2 or "infeasible" in str(res.message).lower()) else f"other ({res.status})")
    return out


# ---------- states, groups, replay ----------

def sample_states(dv: list[dict], n_first: int = 60, n_later: int = 140, seed: int = 0) -> list[int]:
    """The fixed sample (plan): n_first dev states with an empty history and n_later with a history, drawn with rng(seed)."""
    rng = np.random.default_rng(seed)
    first = [i for i, r in enumerate(dv) if not r["prev"]]
    later = [i for i, r in enumerate(dv) if r["prev"]]
    return sorted(rng.choice(first, size=n_first, replace=False).tolist() + rng.choice(later, size=n_later, replace=False).tolist())


def groups_for(r: dict, props: dict, blist: list[str]) -> list[dict]:
    """The rule groups that apply in state r: P_first with an empty history; otherwise one group per missing prerequisite A
    (the buttons B with a rule (A, B))."""
    idx = {b: i for i, b in enumerate(blist)}
    prev = set(r["prev"])
    if not prev:
        allowed = [idx[a] for a in props["allowed_first"]]
        group = [idx[b] for b in props["forbidden_first"]]
        return [{"group": "P_first", "buttons": group, "against": {B: allowed for B in group}}]
    out = []
    for a in sorted({a for a, _ in props["prec"]}):
        if a in prev:
            continue
        group = sorted(idx[b] for a2, b in props["prec"] if a2 == a)
        out.append({"group": f"P_prec({a}, *)", "buttons": group, "against": {B: [k for k in range(B_) if k != B] for B in group}})
    return out


def forbidden_in(r: dict, b: str, props: dict) -> bool:
    prev = set(r["prev"])
    if not prev:
        return b not in props["allowed_first"]
    return any(b == bb and a not in prev for a, bb in props["prec"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", required=True, choices=["model", "contains", "milp", "run"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--model", default="tm", choices=["tm"])
    ap.add_argument("--kind", default="words", choices=["words", "bits", "single"],
                    help="words: the MILP on word edits; bits: the secondary bit-flip distance; single: every single edit, enumerated (exact, no solver)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    tmod = TextModel()
    print(f"candidates: {len(tmod.cand)} words (in >= {CAND_MIN_UTT} train customer utterances)", flush=True)
    if args.check == "model":
        rng = np.random.default_rng(0)
        pick = rng.choice(len(tmod.dv), size=args.n, replace=False)
        X0 = tmod.view.transform([tmod.dv[i] for i in pick], {f: v[pick] for f, v in tmod.em["dev"].items()})
        zero_ok = 0
        rand_ok = {"exact": 0, "text_bits_differing": [], "cases": 0}
        for n, i in enumerate(pick):
            r = tmod.dv[i]
            s = tmod.split(r)
            mb = tmod.model_bits(r, [True] * len(s["occ"]), [])
            zero_ok += int((mb[N_MEM:] == X0[n, N_MEM:]).all())
            keep = [bool(rng.random() > 0.3) for _ in s["occ"]]
            ins = list(rng.choice(tmod.cand, size=int(rng.integers(0, 4)), replace=False))
            if not any(keep) and not ins:
                ins = [tmod.cand[0]]
            r2 = tmod.edited_row(r, keep, ins)
            real = tmod.view.transform([r2], {f: v[i : i + 1] for f, v in tmod.em["dev"].items()})[0]
            mb2 = tmod.model_bits(r, keep, ins)
            diff = int((mb2[N_MEM:] != real[N_MEM:]).sum())
            rand_ok["cases"] += 1
            rand_ok["exact"] += int(diff == 0)
            if diff:
                rand_ok["text_bits_differing"].append(diff)
        rec = {"tag": args.tag, "check": "text_model", "n": int(args.n), "zero_edit_exact": zero_ok, "random_edit_exact": rand_ok["exact"],
               "random_cases": rand_ok["cases"], "random_diffs": rand_ok["text_bits_differing"][:20], "n_cand": len(tmod.cand),
               "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        print(json.dumps(rec), flush=True)
        with OUT.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        return
    if args.check == "contains":
        # no solver: with a random real edit set, the real outcome (edited bits, clause values, the model's choice) satisfies
        # every row of the MILP for the group that contains the model's choice
        from dl import cpu_tm_fast  # noqa: E402
        rng = np.random.default_rng(2)
        tm = load_tm(args.seed)
        pick = rng.choice(len(tmod.dv), size=args.n, replace=False)
        stats = {"cases": 0, "fixed_bits_match": 0, "rows_ok": 0, "broken": []}
        for i in pick:
            r = tmod.dv[i]
            s = tmod.split(r)
            keep = [bool(rng.random() > 0.3) for _ in s["occ"]]
            ins = list(rng.choice(tmod.cand, size=int(rng.integers(0, 4)), replace=False))
            if not any(keep) and not ins:
                ins = [tmod.cand[0]]
            r2 = tmod.edited_row(r, keep, ins)
            x2 = tmod.view.transform([r2], {f: v[i : i + 1] for f, v in tmod.em["dev"].items()})[0]
            x0 = tmod.view.transform([r], {f: v[i : i + 1] for f, v in tmod.em["dev"].items()})[0]
            win = int(cpu_tm_fast.class_sums(tm["pack"], x2[None])[0].argmax())
            res = solve_group(tmod, tm, r, x0, [win], {win: list(range(B_))}, 0, 1, real_point=(keep, ins, x2, win))
            stats["cases"] += 1
            stats["fixed_bits_match"] += int(res["fixed_bits_match"])
            stats["rows_ok"] += int(res["rows_broken"] == 0)
            if res["rows_broken"]:
                stats["broken"].append({"state": int(i), "n": res["rows_broken"], "types": res["broken_types"], "in_recent": s["in_recent"],
                                        "u": s["u"][:120], "keep": keep, "ins": ins})
        rec = {"tag": args.tag, "check": "milp_contains_real", "seed": args.seed, **stats, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        print(json.dumps(rec), flush=True)
        with OUT.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        return
    if args.check == "milp":
        # pre-check of the MILP: with a real edit set fixed, the model's choice on the edited text can win and its runner-up cannot
        rng = np.random.default_rng(1)
        tm = load_tm(args.seed)
        from dl import cpu_tm_fast  # noqa: E402
        pick = rng.choice(len(tmod.dv), size=args.n, replace=False)
        res_ = []
        for i in pick:
            r = tmod.dv[i]
            s = tmod.split(r)
            keep = [bool(rng.random() > 0.3) for _ in s["occ"]]
            ins = list(rng.choice(tmod.cand, size=int(rng.integers(0, 3)), replace=False))
            if not any(keep) and not ins:
                ins = [tmod.cand[0]]
            r2 = tmod.edited_row(r, keep, ins)
            x2 = tmod.view.transform([r2], {f: v[i : i + 1] for f, v in tmod.em["dev"].items()})[0]
            x0 = tmod.view.transform([r], {f: v[i : i + 1] for f, v in tmod.em["dev"].items()})[0]
            S = cpu_tm_fast.class_sums(tm["pack"], x2[None])[0]
            own, other = int(S.argmax()), int(np.argsort(-S)[1])
            allk = {own: list(range(B_)), other: list(range(B_))}
            r1 = solve_group(tmod, tm, r, x0, [own], allk, 60, args.threads, fix_edits=(keep, set(ins)))
            r2_ = solve_group(tmod, tm, r, x0, [other], allk, 60, args.threads, fix_edits=(keep, set(ins)))
            res_.append((r1["status"], r2_["status"], r1["seconds"], r2_["seconds"], r1["n_rows"], r1["nnz"]))
        rec = {"tag": args.tag, "check": "milp_fixed_edits", "seed": args.seed, "n": int(args.n),
               "own_feasible": sum(a == "optimal" for a, *_ in res_), "runner_up_infeasible": sum(b == "infeasible" for _, b, *_ in res_),
               "max_seconds": max(max(c, d) for _, _, c, d, _, _ in res_), "median_rows": int(np.median([e for *_, e, _ in res_])),
               "median_nnz": int(np.median([f for *_, f in res_])), "solver": {**SOLVER, "threads": args.threads, **HIGHS_OPTS},
               "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        print(json.dumps(rec), flush=True)
        with OUT.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        return
    # runs (plan logs/plan_e71_local.md; frozen before the first one)
    if not (ROOT / "logs" / "plan_e71_local.frozen.json").exists():
        raise SystemExit("runs start only after the plan is frozen (logs/plan_e71_local.md)")
    from dl import cpu_tm_fast  # noqa: E402
    from lineb_counts import properties  # noqa: E402

    props = properties(tmod.blist)
    tm = load_tm(args.seed)
    states = sample_states(tmod.dv)
    X0 = tmod.view.transform([tmod.dv[i] for i in states], {f: v[states] for f, v in tmod.em["dev"].items()})
    rows_dom = domain_rows(tmod.names)
    done_ = {(d.get("tag"), d.get("model"), d.get("seed"), d.get("state"), d.get("group"), d.get("kind"))
             for d in (json.loads(l) for l in OUT.read_text().splitlines())} if OUT.exists() else set()
    model = f"{args.model}"
    for n, i in enumerate(states):
        r, x0 = tmod.dv[i], X0[n]
        own = int(cpu_tm_fast.class_sums(tm["pack"], x0[None])[0].argmax())
        for g in groups_for(r, props, tmod.blist):
            key = (args.tag, model, args.seed, int(i), g["group"], args.kind)
            if key in done_:
                continue
            rec = {"tag": args.tag, "model": model, "seed": args.seed, "kind": args.kind, "state": int(i), "convo_id": r["convo_id"], "k": r["k"],
                   "group": g["group"], "buttons": [tmod.blist[b] for b in g["buttons"]], "own_choice": tmod.blist[own],
                   "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            if own in g["buttons"]:
                rec["outcome"] = "own choice breaks the rule"
            elif args.kind == "single":
                # every single edit (one deletion, keeping at least one word, or one insertion), through the exact text model
                s = tmod.split(r)
                edits = [("delete", o) for o in range(len(s["occ"])) if len(s["occ"]) > 1] + [("insert", w) for w in tmod.cand]
                Xe = np.stack([tmod.model_bits(r, [q != o for q in range(len(s["occ"]))] if kind_ == "delete" else [True] * len(s["occ"]),
                                               [] if kind_ == "delete" else [o], x0=x0) for kind_, o in edits])
                Xe[:, :N_MEM] = x0[:N_MEM]
                ch = cpu_tm_fast.class_sums(tm["pack"], Xe).argmax(1)
                bad = [n_ for n_, c in enumerate(ch) if c in g["buttons"]]
                ex = []
                for n_ in bad[:5]:
                    kind_, o = edits[n_]
                    keep = [q != o for q in range(len(s["occ"]))] if kind_ == "delete" else [True] * len(s["occ"])
                    ins = [] if kind_ == "delete" else [o]
                    r2 = tmod.edited_row(r, keep, ins)
                    x2 = tmod.view.transform([r2], {f: v[i : i + 1] for f, v in tmod.em["dev"].items()})[0]
                    c2 = int(cpu_tm_fast.class_sums(tm["pack"], x2[None])[0].argmax())
                    ex.append({"edit": kind_, "word": s["occ"][o] if kind_ == "delete" else o, "choice": tmod.blist[int(ch[n_])],
                               "replay_choice": tmod.blist[c2], "replay_breaks_rule": bool(c2 in g["buttons"])})
                rec.update(kind="single", n_edits_tried=len(edits), n_violating=len(bad), examples=ex, in_recent=s["in_recent"],
                           outcome="certified at k = 1" if not bad else "a single edit breaks the rule",
                           violating_buttons=dict(Counter(tmod.blist[int(ch[n_])] for n_ in bad)))
            elif args.kind == "bits":
                res = solve_bits(tm, rows_dom, x0, g["buttons"], g["against"], args.time_limit, args.threads)
                x = res.pop("x", None)
                rec.update(res, kind="bits")
                if x is not None:
                    rec["replay_choice"] = tmod.blist[int(cpu_tm_fast.class_sums(tm["pack"], x[None])[0].argmax())]
                    rec["replay_breaks_rule"] = bool(tmod.blist.index(rec["replay_choice"]) in g["buttons"])
                rec["outcome"] = {"optimal": "flips found", "infeasible": "certified for every flip set", "time limit": "bounds only"}.get(res["status"], res["status"])
            else:
                res = solve_group(tmod, tm, r, x0, g["buttons"], g["against"], args.time_limit, args.threads)
                rec.update(res, kind="words")
                if "edits" in res:  # an edit set that makes a forbidden button win: replay it through the real pipeline
                    r2 = tmod.edited_row(r, res["keep"], res["inserted"])
                    x2 = tmod.view.transform([r2], {f: v[i : i + 1] for f, v in tmod.em["dev"].items()})[0]
                    c2 = tmod.blist[int(cpu_tm_fast.class_sums(tm["pack"], x2[None])[0].argmax())]
                    rec.update(replay_choice=c2, replay_breaks_rule=bool(forbidden_in(r, c2, props) and tmod.blist.index(c2) in g["buttons"]),
                               edited_utterance=r2["cust"].split(" | ")[-1][:300])
                rec["outcome"] = {"optimal": "edits found", "infeasible": "certified for every edit set",
                                  "time limit": "bounds only"}.get(res["status"], res["status"])
            with OUT.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps({k: rec.get(k) for k in ("state", "group", "outcome", "edits", "dual_bound_edits", "seconds", "replay_breaks_rule")}), flush=True)


if __name__ == "__main__":
    main()
