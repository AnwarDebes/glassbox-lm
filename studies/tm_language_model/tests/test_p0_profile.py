"""CPU-only tests for experiments/p0_profile.py: the nvprof CSV parser and the grouping of
kernel times. No GPU, no CAIR import."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import p0_profile as profile  # noqa: E402

# Layout of `nvprof --print-gpu-summary --print-api-summary --csv --log-file F`: status
# lines carry a ==PID== prefix, the table has a header row and then a units row.
SAMPLE_ONE_TABLE = """==4242== NVPROF is profiling process 4242, command: python worker.py
==4242== Profiling application: python worker.py
==4242== Profiling result:
"Type","Time(%)","Time","Calls","Avg","Min","Max","Name"
,%,ms,,ms,ms,ms,
"GPU activities",50.000000,200.000000,100,2.000000,1.900000,2.100000,"evaluate"
"GPU activities",25.000000,100.000000,100,1.000000,0.900000,1.100000,"select_clause_updates"
"GPU activities",12.500000,50.000000,100,0.500000,0.400000,0.600000,"update"
"GPU activities",2.500000,10.000000,100,0.100000,0.090000,0.110000,"update_message"
"GPU activities",5.000000,20.000000,100,0.200000,0.190000,0.210000,"calculate_messages_conditional(unsigned int*, int)"
"GPU activities",2.500000,10.000000,100,0.100000,0.090000,0.110000,"calculate_messages"
"GPU activities",2.500000,10.000000,200,0.050000,0.040000,0.060000,"[CUDA memcpy HtoD]"
"API calls",60.000000,300.000000,1000,0.300000,0.010000,2.500000,"cuCtxSynchronize"
"API calls",40.000000,200.000000,700,0.285714,0.020000,0.900000,"cuLaunchKernel"
"""

# The same kind of data as two tables with different units per table.
SAMPLE_TWO_TABLES = """======== Profiling result:
"Time(%)","Time","Calls","Avg","Min","Max","Name"
%,s,,us,us,us,
80.0,0.4,100,4000.0,3900.0,4100.0,"evaluate"
20.0,0.1,100,1000.0,900.0,1100.0,"prepare_messages"

======== API calls:
"Time(%)","Time","Calls","Avg","Min","Max","Name"
%,us,,ns,ns,ns,
100.0,250000.0,700,357142.0,20000.0,900000.0,"cuLaunchKernel"
"""


def test_parse_single_table_with_type_column():
    parsed = profile.parse_nvprof_csv(SAMPLE_ONE_TABLE)
    assert [row["name"] for row in parsed["api"]] == ["cuCtxSynchronize", "cuLaunchKernel"]
    assert len(parsed["gpu"]) == 7
    evaluate = parsed["gpu"][0]
    assert evaluate == {"name": "evaluate", "time_ms": 200.0, "calls": 100, "avg_ms": 2.0,
                        "min_ms": 1.9, "max_ms": 2.1, "percent": 50.0}
    assert sum(row["time_ms"] for row in parsed["gpu"]) == pytest.approx(400.0)


def test_parse_converts_units_per_table():
    parsed = profile.parse_nvprof_csv(SAMPLE_TWO_TABLES)
    assert [row["name"] for row in parsed["gpu"]] == ["evaluate", "prepare_messages"]
    assert parsed["gpu"][0]["time_ms"] == pytest.approx(400.0)
    assert parsed["gpu"][0]["avg_ms"] == pytest.approx(4.0)
    assert parsed["api"][0]["name"] == "cuLaunchKernel"
    assert parsed["api"][0]["time_ms"] == pytest.approx(250.0)
    assert parsed["api"][0]["min_ms"] == pytest.approx(0.02)


def test_parse_of_text_without_tables_is_empty():
    assert profile.parse_nvprof_csv("==1== Warning: nothing profiled\n") == {"gpu": [], "api": []}
    assert profile.parse_nvprof_csv("") == {"gpu": [], "api": []}


def test_bare_kernel_name():
    assert profile.bare_kernel_name("evaluate(unsigned int*, int*)") == "evaluate"
    assert profile.bare_kernel_name("update_message") == "update_message"
    assert profile.bare_kernel_name("[CUDA memcpy HtoD]") == "[CUDA memcpy HtoD]"


def test_split_groups_and_class_loop():
    parsed = profile.parse_nvprof_csv(SAMPLE_ONE_TABLE)
    per_kernel: dict[str, float] = {}
    for row in parsed["gpu"]:
        name = profile.bare_kernel_name(row["name"])
        per_kernel[name] = per_kernel.get(name, 0.0) + row["time_ms"]
    split = profile.split_from_kernel_ms(per_kernel)
    assert split["group_ms"] == {"evaluate": 200.0, "select_clause_updates": 100.0,
                                 "update": 60.0, "calculate_messages": 30.0, "other": 10.0}
    assert split["total_ms"] == pytest.approx(400.0)
    assert split["class_loop_ms"] == pytest.approx(360.0)
    assert split["class_loop_fraction"] == pytest.approx(0.9)
    assert sum(split["group_fraction"].values()) == pytest.approx(1.0)


def test_nvprof_prefix_profiles_only_the_marked_region():
    prefix = profile.nvprof_prefix("/usr/local/cuda/bin/nvprof", Path("/tmp/x_%p.csv"))
    assert prefix[0].endswith("nvprof")
    assert prefix[prefix.index("--profile-from-start") + 1] == "off"
    assert "--csv" in prefix and "--print-gpu-summary" in prefix
    assert prefix[-2:] == ["--log-file", "/tmp/x_%p.csv"]


def test_default_config_matches_the_task():
    config = profile.DEFAULT_CONFIG
    assert config["outputs"] == [1000, 16000]
    assert config["base"]["clauses"] == 2000 and config["base"]["depth"] == 2
    assert config["base"]["grid"] == [16 * 13, 1, 1] and config["base"]["block"] == [128, 1, 1]
    assert config["kill_threshold"] == 0.40
