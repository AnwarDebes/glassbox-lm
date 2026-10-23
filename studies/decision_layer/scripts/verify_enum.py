#!/usr/bin/env python
"""e71 v2 (logs/plan_e71_v2_amendment.md): exhaustive search over word edits of the customer's latest utterance u.

For a real dev state (memory as logged) every edit set of size 1 and 2 is tried:
- deleting one or two of u's words, with u keeping at least one word;
- inserting one or two candidate words at u's end (an ordered pair: the bigram order matters, and a repeated word is allowed);
- or one deletion and one insertion.
Every edit set is scored exactly: the bits are rebuilt from the words, as view.transform builds them (the word-table codes
OR-ed; unigrams and bigrams from the edited token sequence), then cpu_tm_fast's clause firing and the integer votes. So
"no edit set of size <= k makes the agent choose a button the rules forbid" is a proof by exhaustion, with no superset,
and every breaking edit set found is real text through the real decision code.

    verify_enum.py --check pre --n 30          # the enumerator's bits against view.transform on edited texts rebuilt from strings
    verify_enum.py --check time --n 2          # time one state's k = 2 search
    verify_enum.py --check run --seed 1        # all 200 states (after the amendment's freeze)
Results: results/explore/verify_e71.jsonl (kind "enum"). Dev only. Added 2026-10-19 for e71 v2.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(1, str(ROOT / "scripts"))

from dl import cpu_tm_fast  # noqa: E402
from verify_local import OUT, TextModel, forbidden_in, groups_for, sample_states  # noqa: E402
from verify_policy import N_MEM, load_tm  # noqa: E402


class Enumerator:
    def __init__(self, tmod: TextModel):
        self.t = t = tmod
        nc = len(t.cand)
        self.cidx = {w: n for n, w in enumerate(t.cand)}
        self.Ccust = np.zeros((nc, t.F), np.uint8)
        self.Crec = np.zeros((nc, t.F), np.uint8)
        for n, w in enumerate(t.cand):
            cd = t.codes(w)
            self.Ccust[n, t.o_cust + cd["xcust"]] = 1
            self.Crec[n, t.o_recent + cd["xrecent"]] = 1
        self.uni = np.array([t.o_words + t.vocab[w] if (w in t.vocab) else -1 for w in t.cand], np.int64)

    def col(self, a: str | None, b: str | None) -> int:
        if a is None or b is None:
            return -1
        c = self.t.vocab.get(f"{a} {b}")
        return -1 if c is None else self.t.o_words + c

    def base(self, r: dict, x0: np.ndarray, s: dict, keep: list[bool]) -> tuple[np.ndarray, str | None]:
        """The bits with u's words kept where keep is True and nothing inserted yet (no junction to the word after u)."""
        t = self.t
        row = x0.copy()
        kept = [w for w, k in zip(s["occ"], keep) if k]
        cust = t.tokens_code(s["cust_fixed"], "xcust").copy()
        for w in kept:
            cust[t.codes(w)["xcust"]] = True
        row[t.o_cust : t.o_recent] = cust
        last = None
        if s["in_recent"]:
            rec = t.tokens_code(s["recent_fixed"], "xrecent").copy()
            for w in kept:
                rec[t.codes(w)["xrecent"]] = True
            row[t.o_recent : t.F] = rec
            words = np.zeros(t.o_cust - t.o_words, np.uint8)
            for g in s["fixed_uni"] | set(kept):
                if g in t.vocab:
                    words[t.vocab[g]] = 1
            for g in s["fixed_bi"]:
                if g in t.vocab:
                    words[t.vocab[g]] = 1
            seq = ([s["jb"]] if s["jb"] else []) + kept
            for a, b in zip(seq, seq[1:]):
                c = t.vocab.get(f"{a} {b}")
                if c is not None:
                    words[c] = 1
            row[t.o_words : t.o_cust] = words
            last = seq[-1] if seq else None
        return row, last

    def close(self, row: np.ndarray, s: dict, last: str | None) -> np.ndarray:
        """No insertion: the junction bigram (last kept word, the word after u) comes back."""
        out = row.copy()
        if s["in_recent"]:
            c = self.col(last, s["ja"])
            if c >= 0:
                out[c] = 1
        return out

    def insert_all(self, row: np.ndarray, s: dict, last: str | None) -> np.ndarray:
        """[n_cand, F]: row with each candidate appended (one insertion each)."""
        X = row[None, :] | self.Ccust
        if s["in_recent"]:
            X |= self.Crec
            n = np.arange(len(self.t.cand))
            u = self.uni >= 0
            X[n[u], self.uni[u]] = 1
            for k, w in enumerate(self.t.cand):
                for c in (self.col(last, w), self.col(w, s["ja"])):
                    if c >= 0:
                        X[k, c] = 1
        return X

    def one(self, row: np.ndarray, s: dict, last: str | None, w: str) -> tuple[np.ndarray, str]:
        """row with w appended, before a second insertion (no junction to the word after u yet)."""
        k = self.cidx[w]
        out = row | self.Ccust[k]
        if s["in_recent"]:
            out |= self.Crec[k]
            if self.uni[k] >= 0:
                out[self.uni[k]] = 1
            c = self.col(last, w)
            if c >= 0:
                out[c] = 1
            return out, w
        return out, last


def votes(m: dict, Wf: np.ndarray, X: np.ndarray) -> np.ndarray:
    """argmax of the integer votes (float32 sums of integers below 2^24 are exact; ties to the lowest index)."""
    F = cpu_tm_fast.fired(m, X).astype(np.float32)
    S = F @ Wf
    return S.argmax(1)


class Reduced:
    """The TM partially evaluated on a base row whose ON bits stay ON (only insertions follow): a clause with a negated
    literal on an ON bit can never fire and is dropped; a positive literal on an ON bit is true and is dropped. What is
    left reads OFF bits only (the set R). Exact for every row that is the base OR more bits."""

    def __init__(self, m: dict, row: np.ndarray):
        F = m["n_features"]
        lits, starts = m["lits"], m["starts"]
        W = m["w"].astype(np.float64)
        on = row[:F].astype(bool)
        bit = np.where(lits < F, lits, lits - F)
        neg = lits >= F
        lon = on[bit]
        dead = np.maximum.reduceat((neg & lon).astype(np.uint8), starts) > 0
        need = ~(~neg & lon)  # positive literals on ON bits are satisfied
        clause_of = np.repeat(np.arange(len(starts)), np.diff(np.append(starts, len(lits))))
        keep_l = need & ~dead[clause_of]
        n_need = np.bincount(clause_of[keep_l], minlength=len(starts))
        alive = ~dead
        always = alive & (n_need == 0)
        self.const = W[always].sum(0).astype(np.float32)
        act = alive & (n_need > 0)
        sel = keep_l & act[clause_of]
        self.R = np.unique(bit[sel])
        ridx = np.full(F, -1, np.int64)
        ridx[self.R] = np.arange(len(self.R))
        nl = np.where(neg[sel], len(self.R) + ridx[bit[sel]], ridx[bit[sel]])
        cl = clause_of[sel]
        order = np.argsort(cl, kind="stable")
        self.lits = nl[order].astype(np.intp)
        counts = np.bincount(cl, minlength=len(starts))[act]
        self.starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.intp)
        self.W = W[act].astype(np.float32)
        self.n_active = int(act.sum())

    def argmax(self, XR: np.ndarray) -> np.ndarray:
        if self.n_active == 0:
            return np.full(len(XR), int(self.const.argmax()))
        L = np.concatenate([XR, 1 - XR], 1)
        Fd = np.minimum.reduceat(L[:, self.lits], self.starts, axis=1).astype(np.float32)
        return (Fd @ self.W + self.const).argmax(1)


def search(en: Enumerator, m: dict, Wf: np.ndarray, r: dict, x0: np.ndarray, k2: bool) -> dict:
    """Every edit set of size 1 (and 2 if k2) of state r, scored exactly: a list of (edit kind, words, argmax array, size).
    Deletions are rebuilt and scored in full; insertions are scored through a Reduced model of their base row."""
    t, s = en.t, en.t.split(r)
    n_o = len(s["occ"])
    full = [True] * n_o
    row0, last0 = en.base(r, x0, s, full)
    out = {"in_recent": s["in_recent"], "n_occ": n_o, "zero_edit_exact": bool((en.close(row0, s, last0) == x0).all()), "sets": []}

    def insertions(row: np.ndarray, last, second: bool):
        """Every single insertion on row (and, if second, every ordered pair), through a Reduced model of row."""
        red = Reduced(m, row)
        R = red.R
        CR = en.Ccust[:, R] | (en.Crec[:, R] if s["in_recent"] else 0)
        pos = {int(b): q for q, b in enumerate(R)}
        uni = np.array([pos.get(int(c), -1) if c >= 0 else -1 for c in en.uni])

        def word_cols(prev, nxt):
            """[n_cand] columns in R: unigram, bigram (prev, w), bigram (w, nxt), for every candidate w."""
            if not s["in_recent"]:
                return []
            cols = [uni]
            cols.append(np.array([pos.get(en.col(prev, w), -1) for w in t.cand]))
            if nxt is not None:
                cols.append(np.array([pos.get(en.col(w, nxt), -1) for w in t.cand]))
            return cols

        def batch(base_R, cols):
            X = base_R[None, :] | CR
            n = np.arange(len(t.cand))
            for c in cols:
                ok = c >= 0
                X[n[ok], c[ok]] = 1
            return red.argmax(X)

        row_R = row[R].astype(np.uint8)
        res = [("single", None, batch(row_R, word_cols(last, s.get("ja"))))]  # .get: off-window states have no "ja" (fix 2026-10-19)
        if second:
            cols_mid = None
            for k1, w1 in enumerate(t.cand):
                b1 = row_R | CR[k1]
                if s["in_recent"]:
                    for c in (uni[k1], pos.get(en.col(last, w1), -1)):
                        if c >= 0:
                            b1[c] = 1
                res.append(("pair", w1, batch(b1, word_cols(w1, s.get("ja")))))
        return res

    # size 1: one insertion
    ins0 = insertions(row0, last0, second=k2)
    out["sets"].append(("insert", [(w,) for w in t.cand], ins0[0][2], 1))
    # size 1: one deletion (u keeps a word); rebuilt and scored in full
    dels = []
    if n_o >= 2:
        for o in range(n_o):
            keep = [q != o for q in range(n_o)]
            row, last = en.base(r, x0, s, keep)
            dels.append((o, keep, row, last))
        out["sets"].append(("delete", [((o, s["occ"][o]),) for o, *_ in dels],
                            votes(m, Wf, np.stack([en.close(row, s, last) for _, _, row, last in dels])), 1))
    if not k2:
        return out
    # size 2: two insertions, in order (w1 then w2; a repeated word allowed)
    for kind, w1, ch in ins0[1:]:
        out["sets"].append(("insert2", [(w1, w2) for w2 in t.cand], ch, 2))
    # size 2: two deletions (u keeps a word)
    if n_o >= 3:
        pairs = [(a, b) for a in range(n_o) for b in range(a + 1, n_o)]
        rows = []
        for a, b in pairs:
            row, last = en.base(r, x0, s, [q not in (a, b) for q in range(n_o)])
            rows.append(en.close(row, s, last))
        out["sets"].append(("delete2", [((a, s["occ"][a]), (b, s["occ"][b])) for a, b in pairs], votes(m, Wf, np.stack(rows)), 2))
    # size 2: one deletion and one insertion (with a single word, u keeps the inserted one)
    for o, keep, row, last in (dels if n_o >= 2 else [(0, [False], *en.base(r, x0, s, [False]))]):
        ch = insertions(row, last, second=False)[0][2]
        out["sets"].append(("delete+insert", [((o, s["occ"][o]), w) for w in t.cand], ch, 2))
    return out


def classify(res: dict, group: list[int], blist: list[str]) -> dict:
    """Per rule group: the edit sets of size 1 and 2 whose outcome is a button of the group."""
    out = {}
    for size in (1, 2):
        n, bad, ex = 0, 0, []
        for kind, words, ch, sz in res["sets"]:
            if sz != size:
                continue
            n += len(ch)
            hit = np.flatnonzero(np.isin(ch, group))
            bad += len(hit)
            for h in hit[: max(0, 5 - len(ex))]:
                ex.append({"edit": kind, "words": list(words[h]), "choice": blist[int(ch[h])]})
        out[f"k{size}"] = {"edit_sets": n, "breaking": bad, "examples": ex}
    return out


def apply_edit(en: Enumerator, r: dict, kind: str, words: list) -> dict:
    """The real edited conversation state for one edit set (deletions name the occurrence index), for the replay."""
    n_o = len(en.t.split(r)["occ"])
    dele = [w[0] for w in words if isinstance(w, (list, tuple))]
    ins = [w for w in words if isinstance(w, str)]
    keep = [q not in dele for q in range(n_o)]
    return en.t.edited_row(r, keep, ins)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", required=True, choices=["pre", "pre2", "time", "run", "milp35"])
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    tmod = TextModel()
    en = Enumerator(tmod)
    tm = load_tm(args.seed)
    m = tm["pack"]
    Wf = m["w"].astype(np.float32)
    assert np.abs(m["w"]).sum(0).max() < 2**24, "votes too large for exact float32 sums"
    print(f"candidates {len(tmod.cand)}; kept clauses {m['n_clauses_kept']}", flush=True)
    if args.check == "pre":
        rng = np.random.default_rng(3)
        pick = rng.choice(len(tmod.dv), size=args.n, replace=False)
        res = {"states": 0, "single_rows": 0, "single_exact": 0, "double_rows": 0, "double_exact": 0, "argmax_equal": 0, "argmax_rows": 0}
        for i in pick:
            r = tmod.dv[i]
            em = {f: v[i : i + 1] for f, v in tmod.em["dev"].items()}
            x0 = tmod.view.transform([r], em)[0]
            s = tmod.split(r)
            n_o = len(s["occ"])
            row0, last0 = en.base(r, x0, s, [True] * n_o)
            res["zero_edit_exact"] = res.get("zero_edit_exact", 0) + int((en.close(row0, s, last0) == x0).all())
            if n_o >= 2:  # every single deletion
                bd = [en.base(r, x0, s, [q != o for q in range(n_o)]) for o in range(n_o)]
                Xd = np.stack([en.close(row, s, last) for row, last in bd])
                Rd = tmod.view.transform([tmod.edited_row(r, [q != o for q in range(n_o)], []) for o in range(n_o)],
                                         {f: np.repeat(v, n_o, 0) for f, v in em.items()})
                res["single_rows"] += n_o
                res["single_exact"] += int((Xd == Rd).all(1).sum())
            X1 = en.insert_all(row0, s, last0)
            rows = [tmod.edited_row(r, [True] * n_o, [w]) for w in tmod.cand]
            R = tmod.view.transform(rows, {f: np.repeat(v, len(rows), 0) for f, v in em.items()})
            res["single_rows"] += len(rows)
            res["single_exact"] += int((X1 == R).all(1).sum())
            a1, a2 = votes(m, Wf, X1), cpu_tm_fast.class_sums(m, R).argmax(1)
            res["argmax_rows"] += len(a1)
            res["argmax_equal"] += int((a1 == a2).sum())
            # random double edits: one deletion and one insertion, or two insertions in order
            for _ in range(50):
                if n_o >= 2 and rng.random() < 0.4:
                    o = int(rng.integers(n_o))
                    keep = [q != o for q in range(n_o)]
                    w = str(rng.choice(tmod.cand))
                    row, last = en.base(r, x0, s, keep)
                    x = en.insert_all(row, s, last)[en.cidx[w]]
                    real = tmod.view.transform([tmod.edited_row(r, keep, [w])], em)[0]
                else:
                    w1, w2 = (str(v) for v in rng.choice(tmod.cand, size=2))
                    row1, last1 = en.one(row0, s, last0, w1)
                    x = en.insert_all(row1, s, last1)[en.cidx[w2]]
                    real = tmod.view.transform([tmod.edited_row(r, [True] * n_o, [w1, w2])], em)[0]
                res["double_rows"] += 1
                res["double_exact"] += int((x == real).all())
            res["states"] += 1
        rec = {"tag": args.tag, "kind": "enum_precheck", **res, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        print(json.dumps(rec), flush=True)
        with OUT.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        return
    if args.check == "pre2":
        # the Reduced scoring against full rows: every single insertion, and every pair for 5 random first words, per state
        rng = np.random.default_rng(5)
        res = {"states": 0, "single_equal": 0, "single_rows": 0, "pair_equal": 0, "pair_rows": 0}
        for i in rng.choice(len(tmod.dv), size=args.n, replace=False):
            r = tmod.dv[i]
            x0 = tmod.view.transform([r], {f: v[i : i + 1] for f, v in tmod.em["dev"].items()})[0]
            s = tmod.split(r)
            n_o = len(s["occ"])
            row0, last0 = en.base(r, x0, s, [True] * n_o)
            full1 = votes(m, Wf, en.insert_all(row0, s, last0))
            out = search(en, m, Wf, r, x0, k2=False)
            red1 = out["sets"][0][2]
            res["single_rows"] += len(full1)
            res["single_equal"] += int((full1 == red1).sum())
            # pairs: recompute with search(k2=True) only for a few first words, by calling the inner path through a full search is
            # too slow here, so compare full rows for 5 random w1 against the Reduced model built on row0
            red = Reduced(m, row0)
            for w1 in rng.choice(tmod.cand, size=5, replace=False):
                row1, last1 = en.one(row0, s, last0, str(w1))
                Xf = en.insert_all(row1, s, last1)
                res["pair_rows"] += len(Xf)
                res["pair_equal"] += int((votes(m, Wf, Xf) == red.argmax(Xf[:, red.R])).sum())
            res["states"] += 1
        rec = {"tag": args.tag, "kind": "enum_precheck2", **res, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        print(json.dumps(rec), flush=True)
        with OUT.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        return
    if args.check == "time":
        rng = np.random.default_rng(4)
        for i in rng.choice(len(tmod.dv), size=args.n, replace=False):
            r = tmod.dv[i]
            x0 = tmod.view.transform([r], {f: v[i : i + 1] for f, v in tmod.em["dev"].items()})[0]
            t0 = time.time()
            res = search(en, m, Wf, r, x0, k2=True)
            n = sum(len(ch) for *_, ch, _ in res["sets"])
            rec = {"tag": args.tag, "kind": "enum_timing", "state": int(i), "n_occ": res["n_occ"], "in_recent": res["in_recent"],
                   "edit_sets": n, "seconds": round(time.time() - t0, 1), "zero_edit_exact": res["zero_edit_exact"],
                   "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            print(json.dumps(rec), flush=True)
            with OUT.open("a") as f:
                f.write(json.dumps(rec) + "\n")
        return
    if args.check == "milp35":
        # k = 3 and 5 for the groups certified at k = 2 by the enumeration: the decision-version MILP (at most k edits;
        # verify_local's reduced exact encoding with its bigram superset), best effort at the time limit. Infeasible means
        # certified; a feasible edit set is replayed and called realizable only if the violation reproduces.
        if not (ROOT / "logs" / "plan_e71_v2_amendment.frozen.json").exists():
            raise SystemExit("runs start only after the v2 amendment is frozen")
        from verify_local import solve_group  # noqa: E402
        from lineb_counts import properties  # noqa: E402

        props = properties(tmod.blist)
        rows = [json.loads(l) for l in OUT.read_text().splitlines()]
        todo = [d for d in rows if d.get("tag") == args.tag.replace("_k35", "") and d.get("kind") == "enum" and d.get("seed") == args.seed
                and d.get("outcome") == "certified at k = 2"]
        done_ = {(d.get("state"), d.get("group"), d.get("budget")) for d in rows if d.get("tag") == args.tag and d.get("kind") == "milp_k"}
        for d in todo:
            i = d["state"]
            r = tmod.dv[i]
            x0 = tmod.view.transform([r], {f: v[i : i + 1] for f, v in tmod.em["dev"].items()})[0]
            g = next(g for g in groups_for(r, props, tmod.blist) if g["group"] == d["group"])
            for k in (3, 5):
                if (i, d["group"], k) in done_:
                    continue
                res = solve_group(tmod, tm, r, x0, g["buttons"], g["against"], args.time_limit, args.threads, budget=k)
                rec = {"tag": args.tag, "kind": "milp_k", "model": "tm", "seed": args.seed, "state": i, "group": d["group"], "budget": k,
                       **{a: res.get(a) for a in ("status", "seconds", "edits", "inserted", "deleted", "n_rows", "nnz", "nodes", "solver")},
                       "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                if res.get("edits") is not None:
                    r2 = tmod.edited_row(r, res["keep"], res["inserted"])
                    x2 = tmod.view.transform([r2], {f: v[i : i + 1] for f, v in tmod.em["dev"].items()})[0]
                    c2 = int(cpu_tm_fast.class_sums(m, x2[None])[0].argmax())
                    rec.update(replay_choice=tmod.blist[c2], replay_breaks_rule=bool(c2 in g["buttons"]))
                rec["outcome"] = {"infeasible": f"certified at k = {k}", "optimal": f"an edit set of size <= {k} found",
                                  "time limit": "unknown"}.get(res["status"], res["status"])
                with OUT.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(json.dumps({a: rec.get(a) for a in ("state", "group", "budget", "outcome", "seconds", "replay_breaks_rule")}), flush=True)
                if res["status"] != "infeasible":
                    break  # not certified at 3: k = 5 cannot be certified either
        return
    # the run (after the v2 amendment's freeze)
    if not (ROOT / "logs" / "plan_e71_v2_amendment.frozen.json").exists():
        raise SystemExit("runs start only after the v2 amendment is frozen (logs/plan_e71_v2_amendment.md)")
    from lineb_counts import properties  # noqa: E402

    props = properties(tmod.blist)
    states = sample_states(tmod.dv)
    done_ = {(d.get("tag"), d.get("seed"), d.get("state"), d.get("group")) for d in (json.loads(l) for l in OUT.read_text().splitlines())}
    for i in states:
        r = tmod.dv[i]
        x0 = tmod.view.transform([r], {f: v[i : i + 1] for f, v in tmod.em["dev"].items()})[0]
        own = int(cpu_tm_fast.class_sums(m, x0[None])[0].argmax())
        groups = [g for g in groups_for(r, props, tmod.blist) if (args.tag, args.seed, int(i), g["group"]) not in done_]
        if not groups:
            continue
        t0 = time.time()
        res1 = search(en, m, Wf, r, x0, k2=False)
        c1 = {g["group"]: classify(res1, g["buttons"], tmod.blist)["k1"] for g in groups}
        need2 = any(c1[g["group"]]["breaking"] == 0 and own not in g["buttons"] for g in groups)
        res2 = search(en, m, Wf, r, x0, k2=True) if need2 else None
        secs = round(time.time() - t0, 1)
        for g in groups:
            rec = {"tag": args.tag, "kind": "enum", "model": "tm", "seed": args.seed, "state": int(i), "convo_id": r["convo_id"], "k": r["k"],
                   "group": g["group"], "buttons": [tmod.blist[b] for b in g["buttons"]], "own_choice": tmod.blist[own],
                   "in_recent": res1["in_recent"], "n_occ": res1["n_occ"], "zero_edit_exact": res1["zero_edit_exact"], "seconds_state": secs,
                   "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            if own in g["buttons"]:
                rec["outcome"] = "own choice breaks the rule"
            else:
                rec["k1"] = c1[g["group"]]
                if rec["k1"]["breaking"]:
                    rec["outcome"] = "a 1-word edit breaks the rule"
                else:
                    rec["k2"] = classify(res2, g["buttons"], tmod.blist)["k2"]
                    rec["outcome"] = "certified at k = 2" if rec["k2"]["breaking"] == 0 else "certified at k = 1; a 2-word edit breaks the rule"
                # replay the examples through the real pipeline (strings, view.transform, the CPU decision)
                for kk in ("k1", "k2"):
                    for ex in rec.get(kk, {}).get("examples", []):
                        r2 = apply_edit(en, r, ex["edit"], ex["words"])
                        x2 = tmod.view.transform([r2], {f: v[i : i + 1] for f, v in tmod.em["dev"].items()})[0]
                        c2 = int(cpu_tm_fast.class_sums(m, x2[None])[0].argmax())
                        ex["replay_choice"] = tmod.blist[c2]
                        ex["replay_breaks_rule"] = bool(c2 in g["buttons"])
                        ex["edited_utterance"] = r2["cust"].split(" | ")[-1][:200]
            with OUT.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps({k: rec.get(k) for k in ("state", "group", "outcome", "seconds_state")}), flush=True)


if __name__ == "__main__":
    main()
