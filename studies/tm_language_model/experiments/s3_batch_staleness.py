"""s3 batch staleness (docs/05 s3; docs/08 KB3): top-1 and throughput vs batch size B on a fixed
corpus, everything else fixed. B in the config's `batch_sizes`. Default corpus: the tiny LM
(5M tokens, V = 2000, the frozen validation eval set; `--corpus planted` for the synthetic check).
KB3: B = 1024 losing more than 1 point vs B = 1 sets the default B to the largest within 0.5 points.
Top-1 is measured once after the last epoch (docs/05 s3 asks for top-1 vs B, not per-epoch curves).

    scripts/cpu_slot.sh 6 -- scripts/run_logged.py --name s3 --config experiments/configs/s3.json \\
        -- .venv-tm/bin/python -u experiments/s3_batch_staleness.py
"""

from __future__ import annotations

import argparse

import numpy as np

from engine_common import ROOT, EngineConfig, TinyLmTask, left_bigram_top1, load_json, make_engine, measured, top1, train_epochs, write_metrics_or_print


def load_corpus(kind: str, seed: int):
    """Planted-rules corpus: (corpus, eval positions, train cut, mode). The tiny LM is TinyLmTask."""
    if kind == "planted":
        from tmlm.data.synthetic import planted_rules_corpus
        corpus, _ = planted_rules_corpus(vocab=500, n_rules=200, n_tokens=200_000, seed=seed)
        cut = int(corpus.tok.shape[0] * 0.9)
        eval_pos = np.arange(cut + 8, corpus.tok.shape[0])
        return corpus, eval_pos, cut, "causal"
    raise ValueError(kind)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "experiments" / "configs" / "s3.json"))
    ap.add_argument("--corpus", default="tiny_lm", choices=["planted", "tiny_lm"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seeds", default=None, help="comma separated, overrides the config")
    ap.add_argument("--oracle", action="store_true")
    args = ap.parse_args()
    cfg_json = load_json(args.config)
    params = cfg_json.get("params", {})
    epochs = args.epochs or int(params.get("epochs", 4))
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else cfg_json["seeds"]
    base = dict(cfg_json.get("engine", {}))
    metrics = {"epochs": measured(epochs), "corpus": measured(args.corpus)}
    task = None
    if args.corpus == "tiny_lm":
        task = TinyLmTask(vocab=int(base.get("vocab", 2000)), n_tokens=int(params.get("tokens", 5_000_000)), mode=base.get("mode", "masked"))
        metrics.update(task.metrics())
        print(f"tiny LM: {task.train.tok.shape[0]} train tokens, {task.targets.shape[0]} targets; bigram {task.left_bigram_top1:.4f} count_probe {task.count_probe_top1:.4f}", flush=True)
    for seed in seeds:
        if task is None:
            corpus, eval_pos, cut, mode = load_corpus(args.corpus, seed)
            gold = corpus.tok[eval_pos]
            bigram = left_bigram_top1(corpus.tok[:cut], int(corpus.kind_map.shape[1]), eval_pos, corpus.tok)
            metrics[f"seed{seed}.left_bigram_top1"] = measured(bigram)
        for B in params["batch_sizes"]:
            over = dict(base)
            over.update({"seed": seed, "batch": int(B)})
            if task is None:
                over.update({"mode": mode, "vocab": int(corpus.kind_map.shape[1])})
                cfg = EngineConfig(**over)
                eng = make_engine(cfg, corpus, oracle=args.oracle)
                totals = train_epochs(eng, epochs)
                r = {"top1": top1(eng, epochs, eval_pos, gold), "examples_per_s": totals["examples"] / max(1e-9, totals["wall_s"]),
                     "n_seeded": int(totals.get("n_seeded", 0)), "n_recycled": int(totals.get("n_recycled", 0)), "wall_s": totals["wall_s"]}
            else:
                r = task.train_eval(EngineConfig(**over), epochs, oracle=args.oracle)
            for k, v in r.items():
                metrics[f"seed{seed}.B{B}.{k}"] = measured(v, unit="examples/s") if k == "examples_per_s" else measured(v)
            print(f"seed {seed} B={B}: top1 {r['top1']:.4f} {r['examples_per_s']:.0f} examples/s  train wall {r['wall_s']:.0f} s", flush=True)
        b1, b1024 = metrics.get(f"seed{seed}.B1.top1"), metrics.get(f"seed{seed}.B1024.top1")
        if b1 and b1024:
            metrics[f"seed{seed}.KB3_B1024_minus_B1_points"] = measured(100.0 * (b1024["value"] - b1["value"]))
    write_metrics_or_print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
