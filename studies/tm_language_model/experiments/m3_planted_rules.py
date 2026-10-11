"""M3 planted-rules gate (docs/04 M3 ACCEPT: ">= 80% of planted rules recovered"; docs/13 H1.4
context; noise-free first).

A synthetic corpus with `n_rules` two-literal rules `IF t[o1]=a AND t[o2]=b THEN y` is generated
(`tmlm.data.synthetic.planted_rules_corpus`), the C++ engine is trained in causal mode, and a rule
counts as RECOVERED when some alive clause (1) has its top vote on y, (2) includes both premise
literals, and (3) includes no other literal that is false whenever the premises hold (checked on
the corpus: every position where the premises hold satisfies the clause). Also reported: causal
top-1 on held-out positions where a rule applies, vs the left-bigram baseline, and CPU throughput.

Run through the run logger (protocol rule 5):
    scripts/cpu_slot.sh 6 -- scripts/run_logged.py --name planted_rules \\
        --config experiments/configs/planted_rules.json -- .venv-tm/bin/python experiments/m3_planted_rules.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tmlm.common import runlog  # noqa: E402
from tmlm.data.synthetic import planted_rules_corpus  # noqa: E402
from tmlm.engine.config import EngineConfig, lit_pack, offset_seg  # noqa: E402
from tmlm.engine.cpu.cpu_engine import CpuEngine  # noqa: E402


def load_config(path: str | None) -> dict:
    if path is None:
        path = str(ROOT / "experiments" / "configs" / "planted_rules.json")
    return json.loads(Path(path).read_text())


def rule_positions(tok: np.ndarray, doc_start: np.ndarray, rule) -> np.ndarray:
    """Positions p where both premises hold inside p's document."""
    n = tok.shape[0]
    p = np.arange(8, n)
    ok = (tok[p + rule.o1] == rule.a) & (tok[p + rule.o2] == rule.b)
    p = p[ok]
    # same document as the premises: p + o1 must be >= doc start of p
    d = np.searchsorted(doc_start, p, side="right") - 1
    return p[p + rule.o1 >= doc_start[d]]


def recovered(engine: CpuEngine, state: dict, cfg: EngineConfig, corpus, rules) -> tuple[int, int, list[dict]]:
    """Returns (n_strict, n_superset, details). superset: some alive clause has its top vote on y and
    includes both premise literals. strict: additionally fires on every one of up to 64 sampled
    positions where the premises hold (so no included literal is ever false there). Both
    definitions are reported."""
    lit_id, lit_state, vote_cls, vote_w, alive = (state[k] for k in ("lit_id", "lit_state", "vote_cls", "vote_w", "alive"))
    included = [[int(l) for l, s in zip(lit_id[j], lit_state[j]) if s >= 128 and l != 0xFFFFFFFF] for j in range(cfg.n_clauses)]
    top_vote = []
    for j in range(cfg.n_clauses):
        cls, w = vote_cls[j], vote_w[j]
        used = [(int(w[k]), int(cls[k])) for k in range(cfg.v_slots) if cls[k] != 0xFFFF]
        top_vote.append(max(used)[1] if used else -1)
    n_strict, n_superset, details = 0, 0, []
    for r in rules:
        want = {lit_pack(offset_seg(r.o1), r.a, 0), lit_pack(offset_seg(r.o2), r.b, 0)}
        supersets = [j for j in range(cfg.n_clauses) if alive[j] and top_vote[j] == r.y and want <= set(included[j])]
        best = None
        if supersets:
            n_superset += 1
            pos = rule_positions(corpus.tok, corpus.doc_start, r)
            sample = pos[: min(64, pos.shape[0])]
            fired_sets = [set(engine.fired_clauses(0, int(p)).tolist()) for p in sample]
            for j in supersets:
                if sample.shape[0] and all(j in f for f in fired_sets):
                    best = j
                    break
        if best is not None:
            n_strict += 1
        details.append({"rule": r.__dict__, "superset_clauses": supersets[:8], "strict_clause": best})
    return n_strict, n_superset, details


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg_json = load_config(args.config)
    params, eng_over = cfg_json["params"], cfg_json.get("engine", {})
    seeds = cfg_json["seeds"]
    epochs = params.get("epochs", 3)
    metrics: dict[str, dict] = {}
    for seed in seeds:
        corpus, rules = planted_rules_corpus(vocab=eng_over.get("vocab", 500), n_rules=params.get("n_rules", 200),
                                             n_tokens=params.get("n_tokens", 200_000), seed=seed, noise=params.get("noise", 0.0))
        cfg = EngineConfig(**{**eng_over, "mode": "causal", "seed": seed})
        engine = CpuEngine(cfg, corpus=corpus)
        t0 = time.time()
        n_ex = 0
        for e in range(epochs):
            for b in range(engine.n_batches()):
                d = engine.train_batch(e, b)
                n_ex += d["n_examples"]
        wall = time.time() - t0
        state = engine.state()
        n_strict, n_superset, details = recovered(engine, state, cfg, corpus, rules)
        # top-1 on positions where some rule applies, in the last 10% of the corpus (held out by position)
        cut = int(corpus.tok.shape[0] * 0.9)
        pos_all = np.unique(np.concatenate([rule_positions(corpus.tok, corpus.doc_start, r) for r in rules]))
        pos_eval = pos_all[pos_all >= cut][:2000]
        scores = engine.predict_scores(epochs, pos_eval)
        top1 = float((scores.argmax(axis=1) == corpus.tok[pos_eval]).mean()) if pos_eval.size else float("nan")
        # left-bigram baseline on the same positions, counts from the first 90%
        V = cfg.vocab
        tr = corpus.tok[:cut].astype(np.int64)
        big = np.zeros((V, V), dtype=np.int64)
        np.add.at(big, (tr[:-1], tr[1:]), 1)
        pred_bigram = big[corpus.tok[pos_eval - 1].astype(np.int64)].argmax(axis=1)
        top1_bigram = float((pred_bigram == corpus.tok[pos_eval]).mean()) if pos_eval.size else float("nan")
        alive_n = int(state["alive"].sum())
        metrics.update({
            f"seed{seed}.rules_recovered_fraction": {"value": n_superset / len(rules), "label": "measured",
                                                    "note": "superset definition: top vote y and both premises included"},
            f"seed{seed}.rules_recovered": {"value": n_superset, "label": "measured"},
            f"seed{seed}.rules_recovered_strict": {"value": n_strict, "label": "measured",
                                                   "note": "strict definition: superset clause that also fires on all sampled premise positions"},
            f"seed{seed}.top1_rule_positions_heldout": {"value": top1, "label": "measured"},
            f"seed{seed}.top1_left_bigram_same_positions": {"value": top1_bigram, "label": "measured"},
            f"seed{seed}.examples_per_s_cpu": {"value": n_ex / wall, "label": "measured", "unit": "examples/s",
                                                "note": f"{epochs} epochs, {n_ex} examples, {wall:.1f}s, OMP threads {os.environ.get('OMP_NUM_THREADS', '?')}"},
            f"seed{seed}.alive_clauses": {"value": alive_n, "label": "measured"},
        })
        print(f"seed {seed}: recovered superset {n_superset}/{len(rules)} strict {n_strict}/{len(rules)}  top1 {top1:.3f} (bigram {top1_bigram:.3f})  "
              f"{n_ex / wall:.0f} examples/s  alive {alive_n}", flush=True)
        run_dir = os.environ.get("TMLM_RUN_DIR")
        if run_dir:
            Path(run_dir, f"rules_seed{seed}.json").write_text(json.dumps(details, indent=1))
    fracs = [v["value"] for k, v in metrics.items() if k.endswith("rules_recovered_fraction")]
    metrics["rules_recovered_fraction_mean"] = {"value": float(np.mean(fracs)), "label": "measured"}
    metrics["gate_80_percent"] = {"value": bool(np.mean(fracs) >= 0.8), "label": "measured"}
    if os.environ.get("TMLM_RUN_DIR"):
        runlog.write_metrics(metrics)
    else:
        print(json.dumps(metrics, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
