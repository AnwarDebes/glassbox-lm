"""s4 deviation ladder (docs/05 s4; docs/11 section 6; docs/08 KB1, KB2): price every departure
from the published coalesced TM, one flag per rung, on the same corpus.

Rungs (docs/11 section 6, order fixed here):
  D0 dense-equivalent mode (only for tabular tasks; the KB1 gate, experiments/m3_parity_tmu.py)
  D1 owned-class lazy decay      (decay_mode lazy_owned instead of eager_exact)
  D2 sparse votes                (always on in the sparse engine; reported as the same run as D1)
  D3 bounded literal slots with insertion (p_ins, p_neg > 0 vs 0)
  D4 seeding                     (rho_seed > 0 vs 0: without seeding nothing is ever learned in the sparse engine)
  D5 the negative-class rule     (neg_mode: the default of docs/11 section 8c, binomial, vs uniform_binomial)
  D6 anchor index                (always on; the index-vs-brute-force test shows it changes nothing)
  D7 batching                    (B = default vs B = 1)
The sparse engine starts at rung D1 with eager decay, uniform-Binomial negatives, no insertion,
B = 1, seeding on, and each later rung flips one flag toward the default engine, so the last rung
IS the default engine: on the tiny LM its top-1 must reproduce experiments/m3_tiny_lm.py for the
same seed (a replay check). Default corpus: the tiny LM; `--corpus planted` for the synthetic check.
Top-1 is measured once after the last epoch. Also reported: count_probe (docs/08 KB2) and the left
bigram on the same positions.

    scripts/cpu_slot.sh 6 -- scripts/run_logged.py --name s4 --config experiments/configs/s4.json \\
        -- .venv-tm/bin/python -u experiments/s4_deviation_ladder.py
"""

from __future__ import annotations

import argparse

from engine_common import ROOT, EngineConfig, TinyLmTask, count_probe_tables, count_probe_top1, left_bigram_top1, load_json, make_engine, measured, top1, train_epochs, write_metrics_or_print
from s3_batch_staleness import load_corpus

START = {"decay_mode": "eager_exact", "neg_mode": "uniform_binomial", "p_ins": 0.0, "p_neg": 0.0, "batch": 1}
LADDER = [
    ("D1_lazy_decay", {"decay_mode": None}),
    ("D3_insertion", {"p_ins": None, "p_neg": None}),
    ("D5_negatives_rule", {"neg_mode": None}),
    ("D7_batching", {"batch": None}),
]
FLAGS = ("decay_mode", "neg_mode", "p_ins", "p_neg", "batch")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "experiments" / "configs" / "s4.json"))
    ap.add_argument("--corpus", default="tiny_lm", choices=["planted", "tiny_lm"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seeds", default=None, help="comma separated, overrides the config")
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--engine", default=None, choices=[None, "cpu", "cuda", "oracle"], help="cpu (default), cuda (bit-exact M4 engine; run under scripts/gpu_run.sh shared) or oracle")
    ap.add_argument("--rungs", default=None, help="comma separated subset of the rung names to run (default: all); the start state and the flag flips of the skipped rungs are still applied so every rung keeps its definition")
    args = ap.parse_args()
    wanted = set(args.rungs.split(",")) if args.rungs else None
    cfg_json = load_json(args.config)
    params = cfg_json.get("params", {})
    epochs = args.epochs or int(params.get("epochs", 4))
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else cfg_json["seeds"]
    base = dict(cfg_json.get("engine", {}))
    metrics = {"epochs": measured(epochs), "corpus": measured(args.corpus), "engine": measured(args.engine or ("oracle" if args.oracle else "cpu")), "rungs": measured([n for n, _ in LADDER if wanted is None or n in wanted])}
    task = None
    if args.corpus == "tiny_lm":
        task = TinyLmTask(vocab=int(base.get("vocab", 2000)), n_tokens=int(params.get("tokens", 5_000_000)), mode=base.get("mode", "masked"))
        metrics.update(task.metrics())
        print(f"tiny LM: {task.train.tok.shape[0]} train tokens, {task.targets.shape[0]} targets; bigram {task.left_bigram_top1:.4f} count_probe {task.count_probe_top1:.4f}", flush=True)
    for seed in seeds:
        if task is None:
            corpus, eval_pos, cut, mode = load_corpus(args.corpus, seed)
            gold = corpus.tok[eval_pos]
            V = int(corpus.kind_map.shape[1])
            metrics[f"seed{seed}.left_bigram_top1"] = measured(left_bigram_top1(corpus.tok[:cut], V, eval_pos, corpus.tok))
            metrics[f"seed{seed}.count_probe_top1"] = measured(count_probe_top1(corpus.tok, eval_pos, count_probe_tables(corpus.tok[:cut], V, (-1, -2, -3))))
        current = dict(base)
        current["seed"] = seed
        if task is None:
            current.update({"mode": mode, "vocab": V})
        current.update(START)
        prev = None
        for name, flip in LADDER:
            for k, v in flip.items():
                current[k] = base.get(k, EngineConfig.__dataclass_fields__[k].default) if v is None else v
            if wanted is not None and name not in wanted:
                continue
            cfg = EngineConfig(**current)
            if task is None:
                eng = make_engine(cfg, corpus, oracle=args.oracle, engine=args.engine)
                totals = train_epochs(eng, epochs)
                r = {"top1": top1(eng, epochs, eval_pos, gold), "examples_per_s": totals["examples"] / max(1e-9, totals["wall_s"]), "wall_s": totals["wall_s"]}
            else:
                r = task.train_eval(cfg, epochs, oracle=args.oracle, engine=args.engine)
            for k, v in r.items():
                metrics[f"seed{seed}.{name}.{k}"] = measured(v, unit="examples/s") if k == "examples_per_s" else measured(v)
            metrics[f"seed{seed}.{name}.flags"] = measured({k: current[k] for k in FLAGS})
            if prev is not None:
                metrics[f"seed{seed}.{name}.delta_points_vs_previous_rung"] = measured(100.0 * (r["top1"] - prev[1]), note=f"vs {prev[0]}")
            prev = (name, r["top1"])
            print(f"seed {seed} {name}: top1 {r['top1']:.4f} {r['examples_per_s']:.0f} examples/s  train wall {r['wall_s']:.0f} s  {metrics[f'seed{seed}.{name}.flags']['value']}", flush=True)
    write_metrics_or_print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
