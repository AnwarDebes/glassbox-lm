"""M3 tiny LM (docs/04 M3; docs/08 KB2; docs/13 H1.4): the engine on the first 5M BPE-2000 tokens of
WikiText-103 train, evaluated per epoch on the FROZEN validation eval set (subword variant), against
the left bigram and the count probe on the same positions with the same visibility.

KB2: below the bigram means stop and diagnose; above the bigram but below 70% of the count probe
means iterate R3 to R6. The per-epoch curve is also the basis of the epoch budget of the big runs
(docs/13 amendment A2).

    scripts/cpu_slot.sh 6 -- scripts/run_logged.py --name tiny_lm --config experiments/configs/tiny_lm.json \\
        -- .venv-tm/bin/python experiments/m3_tiny_lm.py [--mode masked|causal] [--epochs N] [--seeds 1,2]
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from engine_common import ROOT, EngineConfig, count_probe_tables, count_probe_top1, left_bigram_top1, load_json, make_engine, measured, top1, write_metrics_or_print
from tmlm.data.corpus_store import load_split
from tmlm.data.evalsets import load_evalset
from tmlm.engine.api import CorpusArrays

CORPUS = "wikitext103"


def slice_train(corpus: CorpusArrays, n_tokens: int) -> CorpusArrays:
    """First `n_tokens` tokens of the split, cut at a document boundary; static tables (kind_map,
    stop, sym_freq) stay those of the whole train split (docs/11 section 1)."""
    ds = corpus.doc_start
    d = int(np.searchsorted(ds, n_tokens, side="right")) - 1
    end = int(ds[max(1, d)])
    return CorpusArrays(tok=np.ascontiguousarray(corpus.tok[:end]), doc_start=np.ascontiguousarray(ds[:d + 1] if ds[d] == end else ds[:d + 2]),
                        kind_map=np.ascontiguousarray(corpus.kind_map), stop=np.ascontiguousarray(corpus.stop),
                        sym_freq=np.ascontiguousarray(corpus.sym_freq), targets=np.arange(end, dtype=np.uint32))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "experiments" / "configs" / "tiny_lm.json"))
    ap.add_argument("--mode", default=None, choices=[None, "masked", "causal"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seeds", default=None, help="comma separated, overrides the config")
    ap.add_argument("--oracle", action="store_true")
    args = ap.parse_args()
    cfg_json = load_json(args.config)
    eng_over = dict(cfg_json.get("engine", {}))
    if args.mode:
        eng_over["mode"] = args.mode
    mode = eng_over.get("mode", "masked")
    vocab = int(eng_over.get("vocab", 2000))
    n_tokens = int(cfg_json["params"].get("tokens", 5_000_000))
    epochs = args.epochs or int(cfg_json["params"].get("epochs", 6))
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else cfg_json["seeds"]

    train_full = load_split(CORPUS, vocab, "train")
    train = slice_train(train_full, n_tokens)
    val = load_split(CORPUS, vocab, "validation")
    ev = load_evalset(CORPUS, vocab, "validation", "subword")
    hidden = ev["hidden"].astype(np.uint8) if mode == "masked" else None
    targets = ev["targets"].astype(np.uint32)
    gold = val.tok[targets]
    offsets = (-1, -2, -3) if mode == "causal" else (-3, -2, -1, 1, 2, 3)
    tables = count_probe_tables(train.tok, vocab, offsets)
    probe_acc = count_probe_top1(val.tok, targets, tables, hidden)
    bigram_acc = left_bigram_top1(train.tok, vocab, targets, val.tok)
    metrics = {
        "train_tokens": measured(int(train.tok.shape[0])), "train_docs": measured(int(train.doc_start.shape[0] - 1)),
        "val_targets": measured(int(targets.shape[0])), "evalset_sha256": measured(ev["meta"].get("sha256")),
        "left_bigram_top1": measured(bigram_acc, note="true left token, counts from the 5M-token slice"),
        "count_probe_top1": measured(probe_acc, note=f"offsets {offsets}, visible context only"),
        "mode": measured(mode),
    }
    print(f"train {train.tok.shape[0]} tokens, {targets.shape[0]} validation targets; bigram {bigram_acc:.4f} count_probe {probe_acc:.4f}", flush=True)
    for seed in seeds:
        cfg = EngineConfig(**{**eng_over, "seed": seed, "vocab": vocab})
        eng = make_engine(cfg, train, oracle=args.oracle)
        probe = make_engine(cfg, val, oracle=args.oracle)
        curve = []
        for e in range(epochs):
            t0 = time.time()
            n_ex = 0
            diag = {}
            for b in range(eng.n_batches()):
                d = eng.train_batch(e, b)
                n_ex += d["n_examples"]
                for k, v in d.items():
                    diag[k] = diag.get(k, 0) + v
            wall = time.time() - t0
            probe.load_state(eng.state())
            acc = top1(probe, epochs, targets, gold, hidden)
            curve.append(acc)
            alive = int(eng.state()["alive"].sum())
            metrics[f"seed{seed}.epoch{e + 1}.top1"] = measured(acc)
            metrics[f"seed{seed}.epoch{e + 1}.examples_per_s"] = measured(n_ex / wall, unit="examples/s")
            metrics[f"seed{seed}.epoch{e + 1}.alive_clauses"] = measured(alive)
            metrics[f"seed{seed}.epoch{e + 1}.candidates_per_example"] = measured(diag.get("n_candidates", 0) / max(1, n_ex))
            metrics[f"seed{seed}.epoch{e + 1}.fired_per_example"] = measured(diag.get("n_fired", 0) / max(1, n_ex))
            print(f"seed {seed} epoch {e + 1}: top1 {acc:.4f}  {n_ex / wall:.0f} ex/s  alive {alive}  cand/ex {diag.get('n_candidates', 0) / max(1, n_ex):.1f}  fired/ex {diag.get('n_fired', 0) / max(1, n_ex):.2f}", flush=True)
        best = max(curve)
        metrics[f"seed{seed}.best_top1"] = measured(best)
        metrics[f"seed{seed}.KB2_beats_bigram"] = measured(bool(best > bigram_acc))
        metrics[f"seed{seed}.KB2_ratio_to_count_probe"] = measured(best / max(1e-9, probe_acc))
    write_metrics_or_print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
