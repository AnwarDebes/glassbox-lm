"""s5 clause-shard equivalence (docs/05 s5; docs/11 section 9; docs/13 H3.5): the tiny LM trained
unsharded and as 2 and 4 shards with only the three exchanges of section 9. Pass = equal `state_hash`
of the merged shard states and equal DIAG counters after every step. Exchange volumes are logged
against the bound of section 9.

Oracle part (M3): `steps_oracle` steps on the oracle:

    scripts/cpu_slot.sh 1 -- scripts/run_logged.py --name s5_oracle --config experiments/configs/s5.json \\
        --no-gpu-sampler -- .venv-tm/bin/python -u experiments/s5_shard_equivalence.py

GPU part (M4, `--gpu`): `steps_gpu` steps at every B of `gpu_batches` with 1, 2 and 4 `CudaEngine`
shards on the one V100 (`tmlm.engine.cuda.shard.ShardedCudaEngine`), hash and DIAG checked after every
step against the unsharded CudaEngine:

    scripts/gpu_run.sh shared -- scripts/run_logged.py --name s5_gpu --config experiments/configs/s5.json \\
        -- .venv-tm/bin/python -u experiments/s5_shard_equivalence.py --gpu
"""

from __future__ import annotations

import argparse
import time

from engine_common import ROOT, EngineConfig, TinyLmTask, load_json, measured, write_metrics_or_print
from tmlm.engine.oracle.oracle import OracleEngine
from tmlm.engine.oracle.shard import ShardedOracle


def load_train_slice(vocab: int, n_tokens: int):
    """The tiny-LM train slice without the evaluation set and the baselines (the GPU part trains only)."""
    from tmlm.data.corpus_store import load_split
    from m3_tiny_lm import slice_train
    return slice_train(load_split("wikitext103", vocab, "train"), n_tokens)


def gpu_part(cfg_json: dict, args) -> int:
    import time as _time
    from tmlm.engine.cuda.cuda_engine import CudaEngine
    from tmlm.engine.cuda.shard import ShardedCudaEngine
    params = cfg_json["params"]
    steps = args.steps or int(params["steps_gpu"])
    base = dict(cfg_json["engine"])
    vocab = int(base.get("vocab", 2000))
    train = load_train_slice(vocab, int(args.tokens or params.get("tokens", 5_000_000)))
    shards = [S for S in params["shards"] if S > 1]
    metrics = {"part": measured("gpu"), "steps": measured(steps), "train_tokens": measured(int(train.tok.shape[0])),
               "shards": measured(params["shards"]), "batches": measured(params["gpu_batches"]), "hash_checked_every_step": measured(True)}
    all_ok = True
    for seed in cfg_json["seeds"]:
        for B in params["gpu_batches"]:
            cfg = EngineConfig(**{**base, "seed": seed, "batch": int(B)})
            one = CudaEngine(cfg, corpus=train)
            many = {S: ShardedCudaEngine(cfg, train, S) for S in shards}
            n_batches = one.n_batches()
            key = f"seed{seed}.B{B}"
            ok = True
            first_mismatch = None
            t0 = _time.time()
            wall_one = 0.0
            wall_many = {S: 0.0 for S in shards}
            d1 = {}
            for t in range(steps):
                e, b = divmod(t, n_batches)
                t1 = _time.time()
                d1 = one.train_batch(e, b)
                one.synchronize()
                wall_one += _time.time() - t1
                h1 = one.state_hash()
                for S, sh in many.items():
                    t1 = _time.time()
                    d2 = sh.train_batch(e, b)
                    sh.shards[0].synchronize()
                    wall_many[S] += _time.time() - t1
                    same = d1 == d2 and sh.state_hash() == h1 and sh.shards_agree_on_shared_state()
                    if not same and first_mismatch is None:
                        first_mismatch = {"step": t, "shards": S, "diag_equal": d1 == d2,
                                          "diag_diff": {k: (d1[k], d2[k]) for k in d1 if d1[k] != d2[k]},
                                          "hash_equal": sh.state_hash() == h1, "shared_state_agree": sh.shards_agree_on_shared_state()}
                    ok &= same
                if t % 10 == 0 or t == steps - 1:
                    print(f"seed {seed} B={B} step {t + 1}/{steps}: {'equal' if ok else 'MISMATCH'}  {_time.time() - t0:.0f} s  alive {int(one.state()['alive'].sum())}  fired/ex {d1['n_fired'] / max(1, d1['n_examples']):.1f}", flush=True)
                if not ok:
                    break
            all_ok &= ok
            metrics[f"{key}.all_steps_equal"] = measured(ok)
            metrics[f"{key}.first_mismatch"] = measured(first_mismatch)
            metrics[f"{key}.steps_run"] = measured(t + 1)
            metrics[f"{key}.final_state_hash"] = measured(one.state_hash())
            metrics[f"{key}.alive_clauses"] = measured(int(one.state()["alive"].sum()))
            metrics[f"{key}.unsharded_train_wall_s"] = measured(wall_one, note="train_batch calls only, hashing excluded")
            for S, sh in many.items():
                ex = sh.exchange
                per_batch = {k: v / max(1, ex["batches"]) for k, v in ex.items() if k != "batches"}
                metrics[f"{key}.S{S}.train_wall_s"] = measured(wall_many[S], note="S engine instances in sequence on one GPU, hashing excluded")
                metrics[f"{key}.S{S}.exchange_per_batch"] = measured(per_batch, note="entries and bytes per batch, summed over the run then divided by batches; exchange 1 entries = non-zero partial score entries plus S cover counts per example")
                n_ex_total = ex["batches"] * cfg.batch
                fired_classes_per_example = ex["fired_classes"] / max(1, n_ex_total)
                bound = sh.exchange_bound_entries(fired_classes_per_example=fired_classes_per_example)
                per_shard = per_batch["ex1_entries"] / S
                metrics[f"{key}.S{S}.exchange1_entries_per_batch_vs_bound"] = measured(
                    {"measured_entries_all_shards": per_batch["ex1_entries"], "measured_entries_per_shard": per_shard,
                     "fired_classes_per_example": fired_classes_per_example, "fired_clauses_per_example_last_batch": d1.get("n_fired", 0) / max(1, cfg.batch),
                     "bound_entries_per_shard": bound, "within_bound_per_shard": per_shard <= bound},
                    note="bound = B * (K_NEG + 1 + |fired classes| per example, measured as non-zero summed scores) + S + S * MIN_FREE [derived, docs/11 section 9]; a shard sends its non-zero partial entries, a subset of the fired classes, so the per-shard volume is what the bound limits. The earlier proxy (fired clauses for fired classes) understates |fired classes| about 10x because a fired clause votes on up to V_SLOTS classes")
            print(f"seed {seed} B={B}: all steps equal {ok}; unsharded {wall_one:.1f} s, " + ", ".join(f"S{S} {wall_many[S]:.1f} s" for S in shards), flush=True)
            del one, many
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
    metrics["all_equal"] = measured(all_ok)
    write_metrics_or_print(metrics)
    return 0 if all_ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "experiments" / "configs" / "s5.json"))
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--tokens", type=int, default=None, help="train slice size (default: the tiny LM's 5M)")
    ap.add_argument("--gpu", action="store_true", help="the M4 part: CudaEngine shards, steps_gpu steps at every B of gpu_batches")
    args = ap.parse_args()
    cfg_json = load_json(args.config)
    if args.gpu:
        return gpu_part(cfg_json, args)
    params = cfg_json["params"]
    steps = args.steps or int(params["steps_oracle"])
    base = dict(cfg_json["engine"])
    task = TinyLmTask(vocab=int(base.get("vocab", 2000)), n_tokens=int(args.tokens or params.get("tokens", 5_000_000)), mode=base.get("mode", "masked"))
    metrics = {"steps": measured(steps), "train_tokens": measured(int(task.train.tok.shape[0])), "shards": measured(params["shards"])}
    for seed in cfg_json["seeds"]:
        cfg = EngineConfig(**{**base, "seed": seed})
        one = OracleEngine(cfg, corpus=task.train)
        many = {S: ShardedOracle(cfg, task.train, S) for S in params["shards"] if S > 1}
        n_batches = one.n_batches()
        all_equal = True
        first_mismatch = None
        t0 = time.time()
        fired_classes = 0
        for t in range(steps):
            e, b = divmod(t, n_batches)
            d1 = one.train_batch(e, b)
            h1 = one.state_hash()
            fired_classes += d1.get("n_fired", 0)
            for S, sh in many.items():
                d2 = sh.train_batch(e, b)
                ok = d1 == d2 and sh.state_hash() == h1 and sh.shards_agree_on_shared_state()
                if not ok and first_mismatch is None:
                    first_mismatch = {"step": t, "shards": S, "diag_equal": d1 == d2}
                all_equal &= ok
            if t % 10 == 0 or t == steps - 1:
                print(f"seed {seed} step {t + 1}/{steps}: {'equal' if all_equal else 'MISMATCH'}  {time.time() - t0:.0f} s  alive {sum(one.alive)}", flush=True)
            if not all_equal:
                break
        wall = time.time() - t0
        metrics[f"seed{seed}.all_steps_equal"] = measured(all_equal)
        metrics[f"seed{seed}.first_mismatch"] = measured(first_mismatch)
        metrics[f"seed{seed}.wall_s"] = measured(wall)
        metrics[f"seed{seed}.final_state_hash"] = measured(one.state_hash())
        metrics[f"seed{seed}.alive_clauses"] = measured(int(sum(one.alive)))
        for S, sh in many.items():
            ex = sh.exchange
            per_batch = {k: v / max(1, ex["batches"]) for k, v in ex.items() if k not in ("batches", "examples", "fired_classes")}
            metrics[f"seed{seed}.S{S}.exchange_per_batch"] = measured(per_batch, note="entries and bytes per batch, summed over the run then divided by batches")
            fired_classes_per_example = ex["fired_classes"] / max(1, ex["examples"])
            bound = sh.exchange_bound_entries(fired_classes_per_example=fired_classes_per_example)
            metrics[f"seed{seed}.S{S}.exchange1_entries_per_batch_vs_bound"] = measured(
                {"measured_entries": per_batch["ex1_entries"], "bound_entries": bound, "fired_classes_per_example": fired_classes_per_example,
                 "within_bound": bool(per_batch["ex1_entries"] <= bound)},
                note="bound = B * (K_NEG + 1 + distinct fired classes per example) + S + S * MIN_FREE [derived, docs/11 section 9]; measured over the run")
        print(f"seed {seed}: all steps equal {all_equal}; wall {wall:.0f} s", flush=True)
    write_metrics_or_print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
