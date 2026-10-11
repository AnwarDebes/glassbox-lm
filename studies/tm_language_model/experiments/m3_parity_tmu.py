"""M3 gate KB1 (docs/08; docs/04 M3 ACCEPT "NoisyXOR + one multi-class task within 2 points of tmu").

The oracle in dense-equivalent mode (docs/11 section 6) is a standard Coalesced Tsetlin Machine
and is compared with tmu's TMCoalescedClassifier (CPU, weighted clauses) on NoisyXOR and on a
four-class task. Protocol (docs/12, fixed before this logged run): same clauses, T, s and epochs;
statistic = mean test accuracy over epochs 4 to 10 and over 3 runs (both implementations
oscillate from epoch to epoch, so single snapshots are noise); band = 2 points. The oracle runs
are seeded (1, 2, 3, the docs/13 list); tmu's CPU path is unseeded C rand(), so its three runs
are three draws. Also reports the same statistic for the C++ engine (bit-exact with the oracle
by tests/test_cpu_engine_parity.py, so it must match the oracle number exactly).

    scripts/cpu_slot.sh 6 -- scripts/run_logged.py --name parity_tmu \\
        --config experiments/configs/parity_tmu.json -- .venv-tm/bin/python experiments/m3_parity_tmu.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tmlm.common import runlog  # noqa: E402
from tmlm.data.synthetic import dense_config, noisy_xor, two_xor_multiclass  # noqa: E402
from tmlm.engine.cpu.cpu_engine import CpuEngine  # noqa: E402
from tmlm.engine.oracle.oracle import OracleEngine  # noqa: E402

EPOCHS, FIRST_SCORED = 10, 4
N_FEATURES, N_CLAUSES, T, S = 12, 40, 15, 3.9


def engine_curve(engine_cls, cfg, train, test):
    eng = engine_cls(cfg, tabular=train)
    probe = engine_cls(cfg, tabular=test)
    accs = []
    for e in range(EPOCHS):
        for b in range(eng.n_batches()):
            eng.train_batch(e, b)
        probe.load_state(eng.state())
        pred = probe.predict_scores(0, np.arange(test.y.shape[0])).argmax(axis=1)
        accs.append(float((pred == test.y).mean()))
    return accs, eng.state_hash()


def tmu_curve(train, test):
    from tmu.models.classification.coalesced_classifier import TMCoalescedClassifier
    tm = TMCoalescedClassifier(N_CLAUSES, T, S, platform="CPU", weighted_clauses=True)
    x_tr, y_tr, x_te = train.x.astype(np.uint32), train.y.astype(np.uint32), test.x.astype(np.uint32)
    accs = []
    for _ in range(EPOCHS):
        tm.fit(x_tr, y_tr)
        accs.append(float((tm.predict(x_te) == test.y).mean()))
    return accs


def scored(curve):
    return float(np.mean(curve[FIRST_SCORED - 1:]))


def main() -> int:
    metrics = {}
    tasks = {
        "xor": (2, noisy_xor(2000, N_FEATURES, noise=0.1, seed=1), noisy_xor(1000, N_FEATURES, noise=0.0, seed=2)),
        "two_xor": (4, two_xor_multiclass(3000, N_FEATURES, noise=0.05, seed=1), two_xor_multiclass(1000, N_FEATURES, noise=0.0, seed=2)),
    }
    within = True
    for name, (n_classes, train, test) in tasks.items():
        oracle, cpu, hashes_equal = [], [], True
        t0 = time.time()
        for seed in (1, 2, 3):
            cfg = dense_config(N_FEATURES, n_classes, N_CLAUSES, T, S, seed=seed)
            o_curve, o_hash = engine_curve(OracleEngine, cfg, train, test)
            c_curve, c_hash = engine_curve(CpuEngine, cfg, train, test)
            hashes_equal &= (o_hash == c_hash) and (o_curve == c_curve)
            oracle.append(scored(o_curve)); cpu.append(scored(c_curve))
            metrics[f"{name}.oracle.seed{seed}.curve"] = {"value": o_curve, "label": "measured"}
        t_engine = time.time() - t0
        t0 = time.time()
        tmu = []
        for run in range(3):
            curve = tmu_curve(train, test)
            tmu.append(scored(curve))
            metrics[f"{name}.tmu.run{run}.curve"] = {"value": curve, "label": "measured"}
        t_tmu = time.time() - t0
        gap = float(np.mean(oracle) - np.mean(tmu))
        within &= gap >= -0.02
        metrics.update({
            f"{name}.oracle.mean_epochs{FIRST_SCORED}to{EPOCHS}_3seeds": {"value": float(np.mean(oracle)), "label": "measured"},
            f"{name}.cpu_engine.mean_epochs{FIRST_SCORED}to{EPOCHS}_3seeds": {"value": float(np.mean(cpu)), "label": "measured"},
            f"{name}.tmu.mean_epochs{FIRST_SCORED}to{EPOCHS}_3runs": {"value": float(np.mean(tmu)), "label": "measured"},
            f"{name}.gap_oracle_minus_tmu": {"value": gap, "label": "measured"},
            f"{name}.cpu_engine_bit_exact_with_oracle": {"value": bool(hashes_equal), "label": "measured"},
            f"{name}.wall_s.oracle_and_cpu": {"value": t_engine, "label": "measured", "unit": "s"},
            f"{name}.wall_s.tmu": {"value": t_tmu, "label": "measured", "unit": "s"},
        })
        print(f"{name}: oracle {np.mean(oracle):.4f} cpu {np.mean(cpu):.4f} tmu {np.mean(tmu):.4f} gap {gap:+.4f} bit_exact {hashes_equal}", flush=True)
    metrics["KB1_within_two_points"] = {"value": bool(within), "label": "measured"}
    metrics["protocol"] = {"value": f"mean test accuracy over epochs {FIRST_SCORED} to {EPOCHS}, 3 runs; clauses {N_CLAUSES}, T {T}, s {S}", "label": "measured"}
    if os.environ.get("TMLM_RUN_DIR"):
        runlog.write_metrics(metrics)
    else:
        print(json.dumps(metrics, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
