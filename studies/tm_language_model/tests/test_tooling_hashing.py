"""tmlm.common.hashing: every digest is rebuilt by hand from the documented byte format,
so an accidental format change (which would orphan all hashes under results/) fails."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tmlm.common import hashing


def test_sha256_file_matches_hashlib_across_chunk_sizes(tmp_path):
    path = tmp_path / "blob.bin"
    data = bytes(range(256)) * 1000
    path.write_bytes(data)
    expected = hashlib.sha256(data).hexdigest()
    assert hashing.sha256_file(path) == expected
    assert hashing.sha256_file(path, chunk_size=7) == expected
    assert hashing.sha256_file(str(path)) == expected
    empty = tmp_path / "empty"
    empty.write_bytes(b"")
    assert hashing.sha256_file(empty) == hashlib.sha256(b"").hexdigest()
    assert hashing.sha256_bytes(b"abc") == hashlib.sha256(b"abc").hexdigest()


def test_sha256_array_follows_the_documented_format():
    arr = np.arange(6, dtype=np.int32).reshape(2, 3)
    expected = hashlib.sha256(
        b"tmlm.ndarray.v1\x00" + b"<i4\x00" + b"2,3\x00" + arr.tobytes()
    ).hexdigest()
    assert hashing.sha256_array(arr) == expected
    scalar = np.array(7, dtype=np.uint16)
    assert hashing.sha256_array(scalar) == hashlib.sha256(
        b"tmlm.ndarray.v1\x00" + b"<u2\x00" + b"\x00" + scalar.tobytes()
    ).hexdigest()


def test_sha256_array_depends_on_dtype_shape_and_values():
    base = np.arange(12, dtype=np.int32)
    digests = {
        hashing.sha256_array(base),
        hashing.sha256_array(base.reshape(3, 4)),
        hashing.sha256_array(base.reshape(4, 3)),
        hashing.sha256_array(base.astype(np.int64)),
        hashing.sha256_array(base.astype(np.uint32)),  # same bytes, other dtype
        hashing.sha256_array(base.view(np.uint8)),  # same bytes, other dtype and shape
        hashing.sha256_array(base + 1),
        hashing.sha256_array(np.empty((0,), dtype=np.int32)),
        hashing.sha256_array(np.empty((0, 5), dtype=np.int32)),
    }
    assert len(digests) == 9


def test_sha256_array_ignores_memory_layout():
    arr = np.arange(24, dtype=np.float32).reshape(4, 6)
    strided = arr[::2, ::3]
    assert not strided.flags["C_CONTIGUOUS"]
    assert hashing.sha256_array(strided) == hashing.sha256_array(strided.copy())
    fortran = np.asfortranarray(arr)
    assert hashing.sha256_array(fortran) == hashing.sha256_array(arr)
    assert hashing.sha256_array(arr.T) != hashing.sha256_array(arr)
    assert hashing.sha256_array(np.array([True, False])) == hashing.sha256_array(
        np.array([True, False])
    )


def test_sha256_array_rejects_unhashable_inputs():
    with pytest.raises(TypeError):
        hashing.sha256_array([1, 2, 3])
    with pytest.raises(TypeError):
        hashing.sha256_array(np.array([{"a": 1}], dtype=object))
    with pytest.raises(TypeError):
        hashing.sha256_array(np.zeros(2, dtype=[("a", np.int32), ("b", np.float32)]))


def test_config_hash_follows_the_documented_format():
    config = {"b": [1, 2.5, "x", None, True], "a": {"z": 1, "y": (3, 4)}}
    text = '{"a":{"y":[3,4],"z":1},"b":[1,2.5,"x",null,true]}'
    assert hashing.canonical_json(config) == text
    assert hashing.config_hash(config) == hashlib.sha256(text.encode("ascii")).hexdigest()


def test_config_hash_is_independent_of_key_order_and_container_type():
    one = {"a": 1, "b": {"c": [1, 2, 3], "d": "x"}, "e": None}
    two = {"e": None, "b": {"d": "x", "c": (1, 2, 3)}, "a": 1}
    assert hashing.config_hash(one) == hashing.config_hash(two)
    assert hashing.config_hash(one) != hashing.config_hash({**one, "a": 2})
    assert hashing.config_hash({"a": 1}) != hashing.config_hash({"a": 1.0})
    assert hashing.config_hash({"a": [1, 2]}) != hashing.config_hash({"a": [2, 1]})
    # non-ASCII text is escaped, so the digest does not depend on any file encoding
    assert hashing.canonical_json({"a": chr(0xE9)}) == '{"a":"' + chr(92) + 'u00e9"}'


def test_config_hash_accepts_numpy_values_and_paths():
    plain = {"n": 3, "lr": 0.5, "shape": [2, 3], "path": "data/x.bin", "flag": True}
    rich = {"n": np.int64(3), "lr": np.float64(0.5), "shape": np.array([2, 3]),
            "path": Path("data/x.bin"), "flag": np.bool_(True)}
    assert hashing.config_hash(rich) == hashing.config_hash(plain)


def test_config_hash_rejects_ambiguous_values():
    for bad in ({1: "int key"}, {"s": {1, 2}}, {"b": b"bytes"}, {"o": object()}):
        with pytest.raises(TypeError):
            hashing.config_hash(bad)
    for bad in ({"x": float("nan")}, {"x": float("inf")}):
        with pytest.raises(ValueError):
            hashing.config_hash(bad)


def test_config_hash_is_stable_across_processes(repo_root):
    """Same digest in a fresh interpreter with a different PYTHONHASHSEED."""
    config = {"vocab": 16000, "clauses": [8192, 65536], "names": {"b": "x", "a": "y"}}
    code = (
        "import sys; sys.path.insert(0, sys.argv[1])\n"
        "from tmlm.common import hashing\n"
        "print(hashing.config_hash({'names': {'a': 'y', 'b': 'x'}, "
        "'clauses': [8192, 65536], 'vocab': 16000}))\n"
    )
    for seed in ("1", "2"):
        out = subprocess.run([sys.executable, "-c", code, str(repo_root / "src")],
                             capture_output=True, text=True, check=True,
                             env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"})
        assert out.stdout.strip() == hashing.config_hash(config)


def test_sha256_tree_follows_the_documented_format(tmp_path):
    (tmp_path / "src/sub").mkdir(parents=True)
    (tmp_path / "src/sub/z.py").write_bytes(b"zz")
    (tmp_path / "src/a.py").write_bytes(b"a")
    (tmp_path / "env").mkdir()
    (tmp_path / "env/e.sh").write_bytes(b"")
    (tmp_path / "src/skip.so").write_bytes(b"binary")
    (tmp_path / "src/__pycache__").mkdir()
    (tmp_path / "src/__pycache__/a.pyc").write_bytes(b"cache")

    files = hashing.iter_tree_files(tmp_path, ["src", "missing", "env"],
                                    ignore_dirs=["__pycache__"], ignore_file_suffixes=[".so"])
    assert files == ["env/e.sh", "src/a.py", "src/sub/z.py"]

    expected = hashlib.sha256()
    expected.update(b"tmlm.tree.v1\x00")
    for rel, data in (("env/e.sh", b""), ("src/a.py", b"a"), ("src/sub/z.py", b"zz")):
        expected.update(len(rel).to_bytes(8, "big") + rel.encode())
        expected.update(len(data).to_bytes(8, "big") + data)
    assert hashing.sha256_tree(tmp_path, ["src", "missing", "env"], ignore_dirs=["__pycache__"],
                               ignore_file_suffixes=[".so"]) == expected.hexdigest()
