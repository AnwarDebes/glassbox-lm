#!/usr/bin/env python
"""e71 v2, after the fix of 2026-10-19 (note logs/plan_e71_v2_note_2026-10-19.md): the pre-checks on the sampled states
whose latest customer utterance u is outside the last 3 turns (off-window; 3 of the 200), which no earlier pre-check
reached. For these states only the customer field (xcust) changes under an edit of u.

    e71_offwindow_precheck.py --seed 1 --tag e71v2_offwindow_precheck

Per off-window sampled state, as verify_enum.py --check pre and pre2 do for random states:
- with no edit, the enumerator's row equals view.transform's;
- every single deletion and insertion, the enumerator's rows equal view.transform on the edited text rebuilt from
  strings, and their argmax equals cpu_tm_fast's;
- 50 random double edits equal view.transform's;
- the fixed search() (the Reduced scoring) gives the same argmax as full scoring for every single insertion, and for
  every pair after 5 random first words.
Rows go to results/explore/verify_e71.jsonl (kind "enum_precheck_offwindow"). Dev only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(1, str(ROOT / "scripts"))

from dl import cpu_tm_fast  # noqa: E402
from verify_enum import Enumerator, Reduced, search, votes  # noqa: E402
from verify_local import OUT, TextModel, sample_states  # noqa: E402
from verify_policy import load_tm  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tag", default="e71v2_offwindow_precheck")
    args = ap.parse_args()
    tmod = TextModel()
    en = Enumerator(tmod)
    m = load_tm(args.seed)["pack"]
    Wf = m["w"].astype(np.float32)
    states = [i for i in sample_states(tmod.dv) if not tmod.split(tmod.dv[i])["in_recent"]]
    print("off-window sampled states:", states, flush=True)
    rng = np.random.default_rng(7)
    for i in states:
        r = tmod.dv[i]
        em = {f: v[i : i + 1] for f, v in tmod.em["dev"].items()}
        x0 = tmod.view.transform([r], em)[0]
        s = tmod.split(r)
        n_o = len(s["occ"])
        res = {"state": int(i), "n_occ": n_o}
        row0, last0 = en.base(r, x0, s, [True] * n_o)
        res["zero_edit_exact"] = bool((en.close(row0, s, last0) == x0).all())
        single_rows = single_exact = 0
        if n_o >= 2:
            bd = [en.base(r, x0, s, [q != o for q in range(n_o)]) for o in range(n_o)]
            Xd = np.stack([en.close(row, s, last) for row, last in bd])
            Rd = tmod.view.transform([tmod.edited_row(r, [q != o for q in range(n_o)], []) for o in range(n_o)],
                                     {f: np.repeat(v, n_o, 0) for f, v in em.items()})
            single_rows += n_o
            single_exact += int((Xd == Rd).all(1).sum())
        X1 = en.insert_all(row0, s, last0)
        R = tmod.view.transform([tmod.edited_row(r, [True] * n_o, [w]) for w in tmod.cand],
                                {f: np.repeat(v, len(tmod.cand), 0) for f, v in em.items()})
        single_rows += len(tmod.cand)
        single_exact += int((X1 == R).all(1).sum())
        a1, a2 = votes(m, Wf, X1), cpu_tm_fast.class_sums(m, R).argmax(1)
        res.update(single_rows=single_rows, single_exact=single_exact, argmax_rows=len(a1), argmax_equal=int((a1 == a2).sum()))
        dbl = 0
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
            dbl += int((x == real).all())
        res.update(double_rows=50, double_exact=dbl)
        # the fixed search(): Reduced scoring against full scoring
        out = search(en, m, Wf, r, x0, k2=False)
        full1 = votes(m, Wf, X1)
        res.update(reduced_single_rows=len(full1), reduced_single_equal=int((full1 == out["sets"][0][2]).sum()))
        red = Reduced(m, row0)
        pr = pe = 0
        for w1 in rng.choice(tmod.cand, size=5, replace=False):
            row1, last1 = en.one(row0, s, last0, str(w1))
            Xf = en.insert_all(row1, s, last1)
            pr += len(Xf)
            pe += int((votes(m, Wf, Xf) == red.argmax(Xf[:, red.R])).sum())
        res.update(reduced_pair_rows=pr, reduced_pair_equal=pe, search_in_recent=out["in_recent"])
        rec = {"tag": args.tag, "kind": "enum_precheck_offwindow", "seed": args.seed, **res,
               "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        print(json.dumps(rec), flush=True)
        with OUT.open("a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
