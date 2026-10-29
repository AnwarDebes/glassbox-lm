"""E2 query bits: the user's query through the C4 counting table + existsT (DESIGN 1.4; the recipe of R21 and E1).
No neural network at any stage. Decision-layer venv (torch on the CPU, numpy):

    CUDA_VISIBLE_DEVICES= .venv-dl/bin/python -I -B scripts/query_bits.py --out-dir data/query_bits

Words: glassbox-lm/studies/text_understanding/w1/tables/ppw_S1000_d500_e0.5.npz (C4 counting + SVD) and its tokenizer
(wordtab_feats.glove_tokens); a word without a live table row is dropped. Per fit set (the training suites of a fold):
spherical k-means (dl.boolean._spherical_kmeans, K concepts, 25 iterations, CPU generator seeded 0) over the table rows
of the fit set's query words; per query and concept, the max cosine over its words (0.0 for a query without words);
dl.boolean.Thermo(n_bins) fitted on the fit set's queries. Every query of every suite is then encoded with that fit
(the encoder never sees a held-out query while it is fitted). One JSON per fit set:
  fold_<held>.json            fit on the 3 other suites (outer fold)
  fold_<held>__inner_<val>.json  fit on the 2 suites left after also holding out <val> (inner fold, for the choices)
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

E2 = Path(__file__).resolve().parents[1]
EMB = Path(f"{_REPO_ROOT}/studies/text_understanding")
TABLE = EMB / "w1" / "tables" / "ppw_S1000_d500_e0.5.npz"
sys.path.insert(0, str(EMB / "w1"))
sys.path.insert(1, str(EMB))
sys.path.insert(2, f"{_REPO_ROOT}/studies/decision_layer/src")
import wordtab_feats as WF
from dl.boolean import Thermo, _spherical_kmeans

SUITES = ["banking", "slack", "travel", "workspace"]


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tasks", default=str(E2 / "data" / "tasks_v1.2.2.json"))
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--bins", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(2)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    tasks = json.loads(Path(args.tasks).read_text())
    prompts = {s: sorted({t["prompt"] for t in tasks[s]["user_tasks"].values()}) for s in SUITES}
    zt = np.load(TABLE)
    words = [str(w) for w in zt["words"]]
    codes = zt["codes"].astype(np.float32)
    live = np.abs(codes).sum(1) > 0
    row_of = {w: i for i, w in enumerate(words) if live[i]}
    En_all = torch.nn.functional.normalize(torch.as_tensor(codes), dim=1)
    all_q = [(s, q) for s in SUITES for q in prompts[s]]
    ids = {q: [row_of[w] for w in WF.glove_tokens(q) if w in row_of] for _, q in all_q}
    toks = {q: WF.glove_tokens(q) for _, q in all_q}
    dropped = sorted({w for q in toks for w in toks[q] if w not in row_of})
    fits = [(h, None) for h in SUITES] + [(h, v) for h in SUITES for v in SUITES if v != h]
    summary = {}
    for held, val in fits:
        fit_suites = [s for s in SUITES if s != held and s != val]
        fit_q = [q for s in fit_suites for q in prompts[s]]
        vocab = sorted({i for q in fit_q for i in ids[q]})
        E = torch.nn.functional.normalize(torch.as_tensor(codes[vocab]), dim=1)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(0)
        C = _spherical_kmeans(E, args.k, 25, gen)

        def exists(qs, C=C):  # C bound now: the concepts of this fit
            out = np.zeros((len(qs), C.shape[0]), np.float32)
            for r, q in enumerate(qs):
                if ids[q]:
                    out[r] = (En_all[ids[q]] @ C.T).max(0).values.numpy()
            return out

        th = Thermo(args.bins).fit(exists(fit_q))
        near = (C @ E.T).topk(min(4, E.shape[0]), dim=1).indices.numpy()
        cname = ["|".join(words[vocab[j]] for j in r) for r in near]
        names, cols = [], []
        for d, t in enumerate(th.th):
            for v in t:
                names.append(f"query ~ concept {d} ({cname[d]}) >= {v:.6g}")
                cols.append((d, float(v)))
        assert len(set(names)) == len(names), "two query bits got the same name (threshold precision)"
        X_all = exists([q for _, q in all_q])
        by_sha = {}
        for r, (_, q) in enumerate(all_q):
            by_sha[sha(q)] = [k for k, (d, v) in enumerate(cols) if X_all[r, d] >= v]
        tag = f"fold_{held}" if val is None else f"fold_{held}__inner_{val}"
        meta = {"fit_suites": fit_suites, "held_out": held, "inner_val": val, "K": int(C.shape[0]), "bins": args.bins,
                "fit_queries": len(fit_q), "fit_vocab": len(vocab), "n_bits": len(names),
                "table": str(TABLE), "table_sha256": hashlib.sha256(TABLE.read_bytes()).hexdigest(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        (out_dir / f"{tag}.json").write_text(json.dumps({"names": names, "by_sha256": by_sha, "meta": meta}))
        on = [len(v) for v in by_sha.values()]
        summary[tag] = {"n_bits": len(names), "fit_vocab": len(vocab), "on_mean": round(float(np.mean(on)), 1)}
        print(tag, summary[tag], flush=True)
    (out_dir / "SUMMARY.json").write_text(json.dumps({"summary": summary, "queries": len(all_q),
                                                       "words_dropped_not_in_table": dropped,
                                                       "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                                                      indent=1))
    print("queries", len(all_q), "dropped words", len(dropped), dropped[:30])


if __name__ == "__main__":
    main()
