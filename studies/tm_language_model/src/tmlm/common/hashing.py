"""Stable SHA-256 hashing of files, directory trees, numpy arrays, and nested configs.

All functions return lowercase hex digests (64 characters). The byte formats are part
of the contract: changing them invalidates every hash stored under results/, so the
tests rebuild each digest by hand from the format described in the docstrings.

numpy is imported lazily so that this module (and tmlm.common.runlog, which uses it)
stays importable when numpy itself is broken.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

_CHUNK = 1 << 20

ARRAY_MAGIC = b"tmlm.ndarray.v1\x00"
TREE_MAGIC = b"tmlm.tree.v1\x00"


def sha256_bytes(data: bytes) -> str:
    """SHA-256 of a bytes object."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | os.PathLike[str], chunk_size: int = _CHUNK) -> str:
    """SHA-256 of a file's contents, read in chunks of `chunk_size` bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_array(arr: Any) -> str:
    """SHA-256 of a numpy array: dtype, shape, and element bytes in C order.

    Format: ARRAY_MAGIC, then `arr.dtype.str` (for example "<i4", byte order included),
    a NUL byte, the shape as comma separated decimals (empty for a 0-d array), a NUL
    byte, then the raw element bytes in C (row major) order. A non-contiguous view
    hashes the same as a contiguous copy with equal values. Arrays that differ in
    dtype or shape hash differently even when their bytes are equal.

    Object and structured dtypes are rejected with TypeError: their bytes are not a
    stable description of their values.
    """
    import numpy as np

    if not isinstance(arr, np.ndarray):
        raise TypeError(f"sha256_array needs a numpy.ndarray, got {type(arr).__name__}")
    if arr.dtype.hasobject or arr.dtype.names is not None:
        raise TypeError(f"sha256_array does not support dtype {arr.dtype!r}")
    digest = hashlib.sha256()
    digest.update(ARRAY_MAGIC)
    digest.update(arr.dtype.str.encode("ascii") + b"\x00")
    digest.update(",".join(str(int(n)) for n in arr.shape).encode("ascii") + b"\x00")
    flat = np.ascontiguousarray(arr).reshape(-1)
    if flat.size:
        digest.update(memoryview(flat.view(np.uint8)))
    return digest.hexdigest()


def normalize_json(obj: Any, *, allow_nan: bool = False) -> Any:
    """Convert `obj` into plain JSON types, or raise.

    Accepted: None, bool, int, float, str, list, tuple (becomes a list), dict with str
    keys, os.PathLike (becomes its string path), numpy scalars and arrays (through
    `.tolist()`). Anything else raises TypeError: sets, bytes, and arbitrary objects
    have no unambiguous JSON form. Non-str dict keys raise TypeError because JSON
    would silently turn the key 1 into "1". NaN and infinities raise ValueError unless
    `allow_nan` is true.
    """
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        if not allow_nan and (obj != obj or obj in (float("inf"), float("-inf"))):
            raise ValueError("non-finite float is not allowed here")
        return obj
    if isinstance(obj, (list, tuple)):
        return [normalize_json(item, allow_nan=allow_nan) for item in obj]
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for key, value in obj.items():
            if not isinstance(key, str):
                raise TypeError(f"dict key {key!r} is not a str")
            out[key] = normalize_json(value, allow_nan=allow_nan)
        return out
    if isinstance(obj, os.PathLike):
        return os.fspath(obj)
    if type(obj).__module__ == "numpy" and hasattr(obj, "tolist"):
        return normalize_json(obj.tolist(), allow_nan=allow_nan)
    raise TypeError(f"value of type {type(obj).__name__} has no stable JSON form")


def canonical_json(obj: Any) -> str:
    """Canonical JSON text of `obj`: sorted keys, no whitespace, ASCII only, no NaN.

    Note that 1 and 1.0 are different values here ("1" against "1.0"), and that a
    tuple equals the list with the same items.
    """
    return json.dumps(
        normalize_json(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def config_hash(config: Any) -> str:
    """SHA-256 of `canonical_json(config)` encoded as ASCII. Independent of key order."""
    return sha256_bytes(canonical_json(config).encode("ascii"))


def iter_tree_files(
    root: str | os.PathLike[str],
    subdirs: Iterable[str],
    ignore_dirs: Iterable[str] = (),
    ignore_dir_suffixes: Iterable[str] = (),
    ignore_file_suffixes: Iterable[str] = (),
) -> list[str]:
    """Sorted POSIX relative paths (relative to `root`) of the files under `subdirs`.

    Missing subdirs are skipped. Symlinked directories are not followed. Broken
    symlinks are skipped. The sort is by the UTF-8 bytes of the relative path, so the
    order does not depend on the locale or on directory listing order.
    """
    root_path = Path(root)
    skip_dirs = set(ignore_dirs)
    dir_suffixes = tuple(ignore_dir_suffixes)
    file_suffixes = tuple(ignore_file_suffixes)
    found: list[str] = []
    for sub in subdirs:
        base = root_path / sub
        if not base.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
            dirnames[:] = [
                d
                for d in dirnames
                if d not in skip_dirs and not (dir_suffixes and d.endswith(dir_suffixes))
            ]
            for filename in filenames:
                if file_suffixes and filename.endswith(file_suffixes):
                    continue
                full = Path(dirpath) / filename
                if not full.is_file():
                    continue
                found.append(full.relative_to(root_path).as_posix())
    found.sort(key=lambda rel: rel.encode("utf-8"))
    return found


def sha256_tree(
    root: str | os.PathLike[str],
    subdirs: Iterable[str],
    ignore_dirs: Iterable[str] = (),
    ignore_dir_suffixes: Iterable[str] = (),
    ignore_file_suffixes: Iterable[str] = (),
) -> str:
    """SHA-256 over sorted relative paths and file contents.

    Format: TREE_MAGIC, then for every file of `iter_tree_files` in order: the length
    of the UTF-8 relative path as 8 bytes big endian, the path bytes, the length of
    the contents as 8 bytes big endian, the contents. The length prefixes make the
    encoding unambiguous, so renaming a file or moving bytes between files changes
    the digest.
    """
    root_path = Path(root)
    digest = hashlib.sha256()
    digest.update(TREE_MAGIC)
    for rel in iter_tree_files(
        root_path, subdirs, ignore_dirs, ignore_dir_suffixes, ignore_file_suffixes
    ):
        rel_bytes = rel.encode("utf-8")
        data = (root_path / rel).read_bytes()
        digest.update(len(rel_bytes).to_bytes(8, "big"))
        digest.update(rel_bytes)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()
