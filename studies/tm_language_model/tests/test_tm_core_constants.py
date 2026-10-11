"""src/tmlm/engine/core/tm_core.h must carry the same constants as src/tmlm/engine/config.py
(docs/11: one spec, three implementations). Also checks the C mix64 against the Python
reference by compiling a tiny probe with g++ when it is available."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from tmlm.engine import config, rng

HEADER = Path(__file__).resolve().parents[1] / "src" / "tmlm" / "engine" / "core" / "tm_core.h"


def _defines() -> dict[str, int]:
    out = {}
    for m in re.finditer(r"^#define TM_(\w+)\s+(0x[0-9A-Fa-f]+|\d+)u?", HEADER.read_text(), re.M):
        out[m.group(1)] = int(m.group(2), 0)
    return out


def test_header_constants_match_config():
    d = _defines()
    assert d["LIT_EMPTY"] == config.LIT_EMPTY
    assert d["ANCHOR_NONE"] == config.ANCHOR_NONE
    assert d["CLS_EMPTY"] == config.CLS_EMPTY
    assert d["N_EXACT_SEG"] == config.N_EXACT_SEG
    assert d["MAX_BAG_K"] == config.MAX_BAG_K
    assert (d["CLUSTER_SIZE_1"], d["CLUSTER_SIZE_2"], d["CLUSTER_SIZE_3"]) == config.CLUSTER_SIZES
    assert d["INCLUDE_THRESHOLD"] == config.INCLUDE_THRESHOLD
    assert d["STATE_INSERT"] == config.STATE_INSERT
    assert d["STATE_SEED"] == config.STATE_SEED
    assert d["STATE_DENSE_INIT"] == config.STATE_DENSE_INIT
    assert d["W_MAX"] == config.W_MAX


def test_header_streams_match_config():
    text = HEADER.read_text()
    for member in config.Stream:
        m = re.search(rf"TM_S_{member.name} = (\d+)", text)
        assert m and int(m.group(1)) == int(member), member


@pytest.mark.skipif(shutil.which("g++") is None, reason="no g++")
def test_c_rng_matches_python_reference(tmp_path):
    probe = tmp_path / "probe.c"
    probe.write_text(
        '#include <stdio.h>\n#include "tm_core.h"\n'
        "int main(void){ printf(\"%llu %u\\n\", (unsigned long long)tm_mix64(0ull), "
        "tm_rng32(7ull, 3ull, 11ull, 5ull, 2ull)); return 0; }\n"
    )
    exe = tmp_path / "probe"
    subprocess.run(["g++", "-x", "c++", "-std=c++17", f"-I{HEADER.parent}", str(probe), "-o", str(exe)], check=True)
    mixed, r32 = subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout.split()
    assert int(mixed) == 0xE220A8397B1DCDAF == rng.mix64(0)
    assert int(r32) == rng.rng32(7, 3, 11, 5, 2)
