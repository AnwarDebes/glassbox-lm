"""CPU-only tests for experiments/p0_compile_16000.py and the scaling harness config.
No GPU, no CAIR import."""

from __future__ import annotations

import sys
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import p0_common as common  # noqa: E402
import p0_compile_16000 as compile16k  # noqa: E402
import p0_scaling_harness as scaling  # noqa: E402


def test_compile_config_is_the_e3_scale_with_the_safe_grid():
    base = compile16k.DEFAULT_CONFIG["base"]
    assert base["outputs"] == 16000
    assert (base["clauses"], base["hypervector_size"], base["depth"],
            base["message_size"]) == (8192, 2048, 3, 1024)
    assert base["grid"] == [16 * 13, 1, 1] and base["block"] == [128, 1, 1]
    assert base["disable_pycuda_cache"] is True
    # Every key of the worker spec must be known to the worker.
    assert set(base) <= set(common.DEFAULT_SPEC)


def test_expected_bytes_match_docs02_memory_formulas():
    sizes = compile16k.expected_bytes(compile16k.DEFAULT_CONFIG["base"])
    # docs/02: ta_chunks = 128 at HV 2048, ta_state = 8192*128*8*4, weights = 16000*8192*4.
    assert sizes["ta_state"] == 8192 * 128 * 8 * 4
    assert sizes["clause_weights"] == 16000 * 8192 * 4
    assert sizes["class_clause_update"] == sizes["clause_weights"]
    assert sizes["message_ta_state"] == 2 * 8192 * 64 * 8 * 4
    assert sizes["encoded_Y"] == 64 * 16000 * 4


def test_scaling_config_covers_the_design_sweep():
    config = scaling.DEFAULT_CONFIG
    series = {entry["id"]: entry for entry in config["series"]}
    assert series["multiclass_depth1"]["outputs"] == [10, 100, 1000, 5000, 16000]
    assert series["multiclass_depth2"]["outputs"] == [10, 100, 1000, 5000, 16000]
    assert series["multiclass_depth2"]["depth"] == 2
    assert series["multioutput_depth1"]["outputs"] == [100, 16000]
    assert config["base"]["clauses"] == 2000 and config["base"]["hypervector_size"] == 256
    assert config["base"]["reps"] >= 5
    assert config["base"]["grid"] == [16 * 13, 1, 1]


def test_characterise_reports_slopes_from_rows():
    rows = [{"status": "ok", "outputs": k, "examples_per_s_median": 1000.0 / k,
             "ms_per_example_median": float(k)} for k in (10, 100, 1000)]
    rows.append({"status": "failed", "outputs": 5000})
    info = scaling.characterise(rows)
    assert info["n_points"] == 3
    assert abs(info["loglog_slope_all"] + 1.0) < 1e-9
    assert abs(info["slowdown_largest_vs_smallest"] - 100.0) < 1e-9
    assert abs(info["linear_fit_ms_per_example"]["per_output_us"] - 1000.0) < 1e-6
