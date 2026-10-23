#!/usr/bin/env python
"""e76 (plan logs/plan_e76_wordedit_c4.md): e71 v2's exhaustive word-edit search on the agent with no neural network at
any stage (R23's tuned ABCD C4 model, tm_mem_c4_s16, seed 1).

    verify_enum_c4.py --check pre      # the enumerator against view.transform (30 random dev states + the 3 off-window)
    verify_enum_c4.py --check pre2     # the Reduced scoring against full scoring (30 random dev states)
    verify_enum_c4.py --check contr    # every dev state whose latest utterance has a contraction: deletions next to it
    verify_enum_c4.py --check run --seed 1   # the 200 sampled states, k = 1 and 2 (after the plan's freeze)

A port, not an edit: scripts/verify_enum.py and verify_local.py are imported unedited.
- TextModelC4: e71's TextModel on the C4 view. dl.count_text.patch is applied (potion's table cannot load), with the
  empty "none" cache. The per-word codes and the text codes use the C4 table's tokenizer. The model is
  results/confirm/R23abcd/tm_mem_c4_s16.
- EnumeratorC4.base: the one change of method. e71 builds the word-table bits after deletions as the fixed text's codes
  OR the kept words' codes. That is exact for potion's WordPiece, but not for the C4 tokenizer, whose contraction rule
  spans word boundaries ("don't" -> "do" + "n't", while e71's word units are "don", "t" and the residue "'"). So after
  deletions the xcust and xrecent bits are rebuilt from the real edited strings (verify_local.edited_row) with the C4
  tokenizer. Insertions are appended after a space, so OR-ing their codes stays exact.
- Everything else is e71 v2's: the 200 states, the 1,626 candidates, the rule groups, the edit sets of size 1 and 2,
  the Reduced scoring of insertions, the outcomes, and the replays through the real pipeline.
Rows go to results/explore/verify_e76.jsonl. Dev only. Added 2026-10-19.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(1, str(ROOT / "scripts"))

import verify_policy as vp  # noqa: E402
from dl import count_text, cpu_tm_fast  # noqa: E402

MODELS = ROOT / "results" / "confirm" / "R23abcd" / "tm_mem_c4_s16"
vp.SRC = MODELS  # load_tm reads model_seed<k>.npz from here

from verify_enum import Enumerator, Reduced, apply_edit, classify, search, votes  # noqa: E402
from verify_local import CAND_MIN_UTT, WORDRE, TextModel, groups_for, sample_states  # noqa: E402

OUT = ROOT / "results" / "explore" / "verify_e76.jsonl"
TABLE = "c4words-S1000-v1"
CONTR_RE = re.compile(r"n't\b|'(s|re|ve|ll|d|m)\b")


class TextModelC4(TextModel):
    def __init__(self):  # TextModel.__init__ with the C4 table, the none cache and R23's model
        self.meta = count_text.patch(TABLE)
        from dl.abcd import ABCDView, load_examples  # noqa: E402

        self.tr, self.blist = load_examples("train")
        self.dv, _ = load_examples("dev")
        E = np.load(ROOT / "data" / "emb" / "abcd__none.npz")
        self.em = {s: {f: E[f"{s}__{f}"] for f in ("cust", "recent", "full")} for s in ("train", "dev")}
        self.view = ABCDView(parts=("mem", "words", "xcust", "xrecent"), lex_k=1000, seed=0).fit(self.tr, self.em["train"], self.blist)
        self.names = self.view.names
        assert self.names == json.loads((MODELS / "bit_names.json").read_text())
        self.tb = count_text.count_table(TABLE)
        self.vocab = self.view.lex.vec.vocabulary_
        self.o_words = vp.N_MEM
        self.o_cust = vp.N_MEM + len(self.vocab)
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
        self._codes, self._tcode = {}, {}
        self.bigrams_with: dict[str, list[str]] = {}
        for g in self.vocab:
            if " " in g:
                for w in set(g.split(" ")):
                    self.bigrams_with.setdefault(w, []).append(g)

    def text_codes(self, text: str, field: str) -> np.ndarray:
        """The field's bits of a whole text, as ExistsTC.transform builds them (an empty text reads as "(nothing yet)")."""
        key = ("full", text, field)
        if key not in self._tcode:
            toks = self.tb.ids([text])[0]
            self._tcode[key] = self.bits[field][toks].any(0) if toks else np.zeros(self.bits[field].shape[1], bool)
        return self._tcode[key]


class EnumeratorC4(Enumerator):
    def base(self, r: dict, x0: np.ndarray, s: dict, keep: list[bool]):
        """e71's base row, with xcust and xrecent rebuilt from the real edited strings (the contraction finding)."""
        row, last = super().base(r, x0, s, keep)
        t = self.t
        r2 = t.edited_row(r, keep, [])
        # an empty text is only possible before an insertion (a single-word u deleted, then a word inserted): no codes then
        for f, lo, hi, key in (("xcust", t.o_cust, t.o_recent, "cust"), ("xrecent", t.o_recent, t.F, "recent")):
            if f == "xrecent" and not s["in_recent"]:
                continue
            txt = r2[key]
            row[lo:hi] = t.text_codes(txt, f) if txt.strip() else np.zeros(hi - lo, bool)
        return row, last


def write(rec: dict) -> None:
    with OUT.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def check_states(tmod, en, m, Wf, idx, rng, n_double: int, contr_only: bool = False, insertions: bool = True) -> dict:
    """The enumerator's rows against view.transform on edited texts rebuilt from strings, and its argmax against cpu_tm_fast."""
    res = {"states": 0, "zero_edit_exact": 0, "single_rows": 0, "single_exact": 0, "argmax_rows": 0, "argmax_equal": 0,
           "double_rows": 0, "double_exact": 0, "contr_rows": 0, "contr_exact": 0}
    for i in idx:
        r = tmod.dv[i]
        em = {f: v[i : i + 1] for f, v in tmod.em["dev"].items()}
        x0 = tmod.view.transform([r], em)[0]
        s = tmod.split(r)
        n_o = len(s["occ"])
        row0, last0 = en.base(r, x0, s, [True] * n_o)
        res["zero_edit_exact"] += int((en.close(row0, s, last0) == x0).all())
        if n_o >= 2:
            keeps = [[q != o for q in range(n_o)] for o in range(n_o)]
            bd = [en.base(r, x0, s, k) for k in keeps]
            Xd = np.stack([en.close(row, s, last) for row, last in bd])
            Rd = tmod.view.transform([tmod.edited_row(r, k, []) for k in keeps], {f: np.repeat(v, n_o, 0) for f, v in em.items()})
            res["single_rows"] += n_o
            res["single_exact"] += int((Xd == Rd).all(1).sum())
            if contr_only:  # every deletion pair, since contractions span word boundaries
                pairs = [(a, b) for a in range(n_o) for b in range(a + 1, n_o)] if n_o >= 3 else []
                for a, b in pairs:
                    k = [q not in (a, b) for q in range(n_o)]
                    row, last = en.base(r, x0, s, k)
                    real = tmod.view.transform([tmod.edited_row(r, k, [])], em)[0]
                    res["contr_rows"] += 1
                    res["contr_exact"] += int((en.close(row, s, last) == real).all())
        if insertions:
            X1 = en.insert_all(row0, s, last0)
            R = tmod.view.transform([tmod.edited_row(r, [True] * n_o, [w]) for w in tmod.cand], {f: np.repeat(v, len(tmod.cand), 0) for f, v in em.items()})
            res["single_rows"] += len(tmod.cand)
            res["single_exact"] += int((X1 == R).all(1).sum())
            a1, a2 = votes(m, Wf, X1), cpu_tm_fast.class_sums(m, R).argmax(1)
            res["argmax_rows"] += len(a1)
            res["argmax_equal"] += int((a1 == a2).sum())
        for _ in range(n_double):
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
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", required=True, choices=["pre", "pre2", "contr", "run"])
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    tmod = TextModelC4()
    en = EnumeratorC4(tmod)
    tm = vp.load_tm(args.seed)
    m = tm["pack"]
    Wf = m["w"].astype(np.float32)
    assert np.abs(m["w"]).sum(0).max() < 2**24, "votes too large for exact float32 sums"
    print(f"candidates {len(tmod.cand)}; kept clauses {m['n_clauses_kept']}; table {tmod.meta['table']}", flush=True)
    base = {"tag": args.tag, "model": "tm_c4_s16", "models": str(MODELS.relative_to(ROOT)), "seed": args.seed,
            "table_file_sha256": tmod.meta["table_file_sha256"], "time": None}
    if args.check == "pre":
        rng = np.random.default_rng(3)
        pick = list(rng.choice(len(tmod.dv), size=args.n, replace=False))
        off = [i for i in sample_states(tmod.dv) if not tmod.split(tmod.dv[i])["in_recent"]]
        res = check_states(tmod, en, m, Wf, pick + off, rng, 50)
        rec = {**base, "kind": "enum_precheck_c4", "off_window_states": off, **res, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        print(json.dumps(rec), flush=True)
        write(rec)
        return
    if args.check == "contr":
        idx = [i for i, r in enumerate(tmod.dv) if r["cust"] and CONTR_RE.search(r["cust"].split(" | ")[-1].lower())]
        rng = np.random.default_rng(6)
        res = check_states(tmod, en, m, Wf, idx, rng, 20, contr_only=True, insertions=False)
        rec = {**base, "kind": "enum_precheck_c4_contractions", "n_states_with_contraction": len(idx), **res,
               "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        print(json.dumps(rec), flush=True)
        write(rec)
        return
    if args.check == "pre2":
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
            res["single_rows"] += len(full1)
            res["single_equal"] += int((full1 == out["sets"][0][2]).sum())
            red = Reduced(m, row0)
            for w1 in rng.choice(tmod.cand, size=5, replace=False):
                row1, last1 = en.one(row0, s, last0, str(w1))
                Xf = en.insert_all(row1, s, last1)
                res["pair_rows"] += len(Xf)
                res["pair_equal"] += int((votes(m, Wf, Xf) == red.argmax(Xf[:, red.R])).sum())
            res["states"] += 1
        rec = {**base, "kind": "enum_precheck2_c4", **res, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        print(json.dumps(rec), flush=True)
        write(rec)
        return
    # the run (after the plan's freeze), as verify_enum.py --check run
    if not (ROOT / "logs" / "plan_e76_wordedit_c4.frozen.json").exists():
        raise SystemExit("runs start only after logs/plan_e76_wordedit_c4.md is frozen")
    from lineb_counts import properties  # noqa: E402

    props = properties(tmod.blist)
    done_ = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            d = json.loads(line)
            done_.add((d.get("tag"), d.get("seed"), d.get("state"), d.get("group")))
    for i in sample_states(tmod.dv):
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
            rec = {**base, "kind": "enum", "state": int(i), "convo_id": r["convo_id"], "k": r["k"], "group": g["group"],
                   "buttons": [tmod.blist[b] for b in g["buttons"]], "own_choice": tmod.blist[own], "in_recent": res1["in_recent"],
                   "n_occ": res1["n_occ"], "zero_edit_exact": res1["zero_edit_exact"], "seconds_state": secs,
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
                for kk in ("k1", "k2"):
                    for ex in rec.get(kk, {}).get("examples", []):
                        r2 = apply_edit(en, r, ex["edit"], ex["words"])
                        x2 = tmod.view.transform([r2], {f: v[i : i + 1] for f, v in tmod.em["dev"].items()})[0]
                        c2 = int(cpu_tm_fast.class_sums(m, x2[None])[0].argmax())
                        ex["replay_choice"] = tmod.blist[c2]
                        ex["replay_breaks_rule"] = bool(c2 in g["buttons"])
                        ex["edited_utterance"] = r2["cust"].split(" | ")[-1][:200]
            write(rec)
            print(json.dumps({k: rec.get(k) for k in ("state", "group", "outcome", "seconds_state")}), flush=True)


if __name__ == "__main__":
    main()
