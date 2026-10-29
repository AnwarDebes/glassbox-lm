#!/usr/bin/env python
"""W1 suite 2 (N7's five sets, CARVED VAL only; the test file is never opened): existsT with a table keyed by
potion-base-32M token ids, then N7's TM setting.

    w1_n7.py --build --dataset trec --table tables/NAME.npz      # feats_n7/NAME/{ds}.npz (train + carved val bits)
    w1_n7.py --dataset trec --table tables/NAME.npz --seed 1     # one TM run on existsT alone
    w1_n7.py --check --dataset trec                              # potion's rows through this path vs N7's cache

Splits and texts: n7_tm.load(dataset, with_test=False) (the 90/10 carve of the published train set,
default_rng(2026)). Bits: n7_tm.exists_bits's potion branch with the table's rows in place of potion's (tokens by
potion's tokenizer file, add_special_tokens=False; a token without a row is dropped): spherical k-means (K 512, 25
iterations, CUDA generator seeded 0) of the distinct train tokens, per text and concept the max cosine over its tokens
(exists_tm.pooled with N7's batch size: 64 for R52 / R8, else 1024), Thermo(8) fitted on train. --check builds the bits
from tables/ref_potion_full.npz and compares them with N7's explore cache (emb_n7/{ds}__explore.npz, existsT_potion).
TM: N7's run loop unchanged in substance (dl/tm.py CoTM, 4,000 clauses, T 80, s 9, 100 epochs, batch 128, hardest
negative, negation, init half; val scored after every epoch, no best-state restore). Statistics: S1-val = mean val
accuracy over epochs 51-100; best val (earliest best epoch). Rows in w1/results_w1_n7val.jsonl.
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))
_HOME = _os.path.expanduser("~")

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EMB = HERE.parent
sys.path.insert(0, str(EMB))
sys.path.insert(0, f"{_REPO_ROOT}/studies/decision_layer/src")
import exists_tm as X  # noqa: E402
import n7_tm as N  # noqa: E402
from dl.boolean import Thermo, _spherical_kmeans  # noqa: E402
from dl.tm import TMConfig  # noqa: E402
from dl.tm_init import make_tm  # noqa: E402

TOK_JSON = Path(f"{_HOME}/.cache/huggingface/hub/models--minishlab--potion-base-32M/snapshots") / X.REV / "tokenizer.json"


def sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def bits_for(table: Path, texts: dict, dev: str, bs: int):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(TOK_JSON))
    z = np.load(table)
    vocab_ids, codes = z["vocab_ids"].astype(np.int64), z["codes"].astype(np.float32)
    has = np.abs(codes).sum(1) > 0
    row_of = {int(t): j for j, t in enumerate(vocab_ids) if has[j]}
    raw = {k: [e.ids for e in tok.encode_batch(t, add_special_tokens=False)] for k, t in texts.items()}
    need = sorted({t for v in raw.values() for x in v for t in x if t in row_of})
    local = {t: j for j, t in enumerate(need)}
    E = torch.as_tensor(codes[[row_of[t] for t in need]], device=dev)
    ids = {k: [[local[t] for t in x if t in local] for x in v] for k, v in raw.items()}
    En = torch.nn.functional.normalize(E, dim=1)
    vocab = sorted({t for x in ids["train"] for t in x})
    gen = torch.Generator(device=dev)
    gen.manual_seed(0)
    Cn = _spherical_kmeans(En[torch.as_tensor(vocab, device=dev)], N.K, 25, gen)
    sims = {k: X.pooled(v, E, En, Cn, dev, bs=bs)["exists"] for k, v in ids.items()}
    th = Thermo(8).fit(sims["train"])
    bits = {k: th.transform(v).astype(np.uint8) for k, v in sims.items()}
    n_tok = {k: sum(len(x) for x in v) for k, v in raw.items()}
    info = {"table": table.stem, "table_sha256": sha16(table), "train_vocab": len(vocab),
            "token_share_without_vector": {k: round(1 - sum(map(len, ids[k])) / max(1, n_tok[k]), 4) for k in ids},
            "texts_without_token": {k: int(sum(1 for x in v if not x)) for k, v in ids.items()},
            "on_share": {k: round(float(b.mean()), 4) for k, b in bits.items()}}
    return bits, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=N.DATASETS)
    ap.add_argument("--table", default="tables/ref_potion_full.npz")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    t0 = time.time()
    dev = "cuda"
    torch.cuda.set_per_process_memory_fraction(0.1)
    table = HERE / args.table
    out_dir = HERE / "feats_n7" / table.stem
    if args.build or args.check:
        sp, C = N.load(args.dataset, False)
        assert set(sp) == {"train", "val"}
        texts = {k: v[0] for k, v in sp.items()}
        bs = 64 if args.dataset in ("r52", "r8") else 1024
        bits, info = bits_for(table, texts, dev, bs)
        if args.check:
            ref = np.load(EMB / "emb_n7" / f"{args.dataset}__explore.npz")
            diff = {k: int((bits[k] != ref[f"existsT_potion_{k}"]).sum()) if bits[k].shape == ref[f"existsT_potion_{k}"].shape else "shape"
                    for k in bits}
            rec = {"check": "W1 suite-2 path vs N7's explore cache (potion rows)", "dataset": args.dataset,
                   "bits_differing": diff, "bits_total": {k: int(b.size) for k, b in bits.items()},
                   "passed": all(v == 0 for v in diff.values())}
            print(json.dumps(rec))
            with open(HERE / "results_w1_checks.jsonl", "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / f"{args.dataset}.npz", **{f"existsT_{k}": b for k, b in bits.items()},
                            **{f"y_{k}": v[1] for k, v in sp.items()})
        info.update({"dataset": args.dataset, "classes": C, "n": {k: len(v[1]) for k, v in sp.items()}, "pooled_bs": bs,
                     "w1_n7_sha256": sha16(Path(__file__).resolve()), "seconds": round(time.time() - t0, 1)})
        (out_dir / f"{args.dataset}.meta.json").write_text(json.dumps(info))
        print(json.dumps(info))
        return
    z = np.load(out_dir / f"{args.dataset}.npz")
    meta = json.loads((out_dir / f"{args.dataset}.meta.json").read_text())
    assert meta["table_sha256"] == sha16(table), "the cache was built from another table"
    Btr, Bva = z["existsT_train"], z["existsT_val"]
    cfg = TMConfig(n_clauses=4000, T=80, s=9.0, epochs=N.EPOCHS, batch_size=128, neg_sampling="hardest", negation=True,
                   max_literals=0, budget_mode="exact", budget_select="threshold", max_negated=0, clause_drop=0.0,
                   half=True, seed=args.seed, eval_every=1)
    tm = make_tm(int(meta["classes"]), Btr.shape[1], cfg, "half")
    Xt = torch.as_tensor(np.ascontiguousarray(Btr), dtype=torch.uint8, device=tm.dev)
    yt = torch.as_tensor(z["y_train"], dtype=torch.long, device=tm.dev)
    Xv = torch.as_tensor(np.ascontiguousarray(Bva), dtype=torch.uint8, device=tm.dev)
    yv = torch.as_tensor(z["y_val"], dtype=torch.long, device=tm.dev)
    curve, n, t1 = [], Xt.shape[0], time.time()
    for epoch in range(cfg.epochs):  # N7's loop: CoTM.fit's batches and random stream, val after every epoch
        perm = torch.randperm(n, generator=tm.gen, device=tm.dev)
        for start in range(0, n, cfg.batch_size):
            bidx = perm[start : start + cfg.batch_size]
            tm._step(Xt[bidx], yt[bidx])
        curve.append(round(float((tm.predict(Xv) == yv).float().mean()) * 100, 2))
    torch.cuda.synchronize()
    best = int(np.argmax(curve))
    rec = {"suite": "W1 suite 2 (N7 sets, carved val)", "dataset": args.dataset, "table": table.stem,
           "table_sha256": meta["table_sha256"], "view": "existsT", "seed": args.seed, "n_bits": int(Btr.shape[1]),
           "val_S1_mean_51_100": round(float(np.mean(curve[-N.LAST:])), 3), "val_at_best": curve[best],
           "best_val_epoch": best + 1, "val_final": curve[-1], "curve": curve, "cfg": asdict(cfg),
           "train_s": round(time.time() - t1, 1), "cache_meta": meta,
           "sha256": {"w1_n7.py": sha16(Path(__file__).resolve()), "n7_tm.py": sha16(EMB / "n7_tm.py")}}
    with open(HERE / "results_w1_n7val.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k not in ("curve", "cfg", "cache_meta")}))


if __name__ == "__main__":
    main()
