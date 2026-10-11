"""Frozen CVTM cache: fitted KN-5 variants, the frozen CVTM-core and per-position arrays
(docs/11 section 11; docs/13 A3 and A4). Written once by `experiments/cvtm_lm.py` (mode
"cache_build", config `experiments/cvtm/configs/cvtm_cache_build.json`) into the run directory:

    <run>/cache/cache_meta.json          layout, dtypes, shapes, sha256 of every file, provenance
    <run>/cache/kn5_all.npz              KN-5 on ALL training documents (the model's suffix clauses)
    <run>/cache/kn5_noH.npz              KN-5 without H  (residue 0; the core's EM inputs)
    <run>/cache/kn5_noHprime.npz         KN-5 without H' (residues 0, 1, 2; the T2 inputs of A4)
    <run>/cache/cvtm_core.npz            the frozen CVTM-core (`tmlm.cvtm.model.CVTMCore.save`)
    <run>/cache/hprime/<name>.npy        every position of the H' documents (3,753,162 at V 8000)
    <run>/cache/validation/<name>.npy    every position of the validation split (282,587)

The .npz files use the integer layout of `tmlm.cvtm.model.kn_to_arrays` (`load_kn` checks every
array hash). Per-position arrays (n positions of the group's stream, d documents; the stream is the
group's documents concatenated in their split order, positions are indices into it):

| name | dtype | shape | content |
|---|---|---|---|
| gold | uint16 | [n] | the token at the position (the causal target) |
| doc_start | int64 | [d + 1] | document boundaries of the group's stream |
| doc_index | int64 | [d] | stable document ID: train index (hprime), validation document number (validation) |
| residue | int8 | [d] | protocol residue of the document (hprime: 0 core_em, 1 t2_train, 2 selection; validation: -1) |
| p_kn_all, p_kn_noH, p_kn_noHp | float64 | [n] | P_KN(gold) of the KN variant (vote form, exp of the float64 vote sum) |
| kstar_all, kstar_noH, kstar_noHp | int8 | [n] | k* under that variant (longest fired suffix context, 0..4) |
| kn_top_ids, kn_top_p | uint16, float64 | [n, 64] | top-64 of P_KN(. given h): noHp variant on hprime, all variant on validation |
| doc_n, doc_g | int32 | [5, n] | n_m and g_m of the document clauses D_0..D_4 (whole-document window) |
| ptr_len, ptr_cand, ptr_pos | int32, int32, int64 | [n] | pointer length L (0 if P does not fire), candidate token (-1), position in the group stream (-1) |
| bucket | int16 | [n] | bucket id (k*, m*, c, l) of the full component set (520 buckets), with the base's k* |
| core_p_gold | float64 | [n] | CVTM-core P(gold) |
| core_top_ids, core_top_p | uint16, float64 | [n, 64] | top-64 of the CVTM-core full distribution |

Base of the core arrays ("core_base" in cache_meta.json): validation uses the frozen CVTM-core (KN on
all training documents, the model that is reported). H' uses the SAME frozen EM weights over the KN
fitted without H' ("kstar_noHp", "p_kn_noHp"), because the frozen core has seen the H' documents and the
T2 inputs of docs/13 A4 use a KN-5 fitted without H'. The EM inputs of the frozen core are reproducible
from the hprime arrays: residue-0 positions, p_kn_noH, kstar_noH, doc_n, doc_g, ptr_len, ptr_cand.
Top-k lists are in descending probability, ties to the lower token id.

Held-out status of every P_KN array (a KN is held out on a position when the position's document was
not among its counting documents): validation, all three variants; hprime residue 0, noH and noHp;
hprime residues 1 and 2, noHp only (p_kn_noH and p_kn_all are in-sample there). The hprime core
arrays use the noHp base everywhere; on residue-0 positions the frozen EM weights were fitted on those
same documents (with the noH base), so they are in-sample for the weights.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from ..common import hashing
from .doc_clauses import DocFeatures
from .mixture import D_ORDERS

FORMAT = "tmlm.cvtm.cache.v1"
GROUPS = ("hprime", "validation")
KN_VARIANTS = {"all": "kn5_all.npz", "noH": "kn5_noH.npz", "noHp": "kn5_noHprime.npz"}
CORE_FILE = "cvtm_core.npz"
META_FILE = "cache_meta.json"
TOPK = 64
ARRAYS = {
    "gold": ("uint16", "[n]", "the token at the position (the causal target)"),
    "doc_start": ("int64", "[d+1]", "document boundaries of the group's stream"),
    "doc_index": ("int64", "[d]", "stable document ID (train index for hprime, validation document number)"),
    "residue": ("int8", "[d]", "protocol residue of the document (hprime 0/1/2; validation -1)"),
    "p_kn_all": ("float64", "[n]", "P_KN(gold), KN-5 on all training documents (vote form)"),
    "p_kn_noH": ("float64", "[n]", "P_KN(gold), KN-5 without H (vote form)"),
    "p_kn_noHp": ("float64", "[n]", "P_KN(gold), KN-5 without H' (vote form)"),
    "kstar_all": ("int8", "[n]", "k* under KN-5 on all training documents"),
    "kstar_noH": ("int8", "[n]", "k* under KN-5 without H"),
    "kstar_noHp": ("int8", "[n]", "k* under KN-5 without H'"),
    "kn_top_ids": ("uint16", "[n,64]", "top-64 classes of P_KN (noHp on hprime, all on validation)"),
    "kn_top_p": ("float64", "[n,64]", "their probabilities"),
    "doc_n": ("int32", "[5,n]", "n_m of D_0..D_4 (whole-document window)"),
    "doc_g": ("int32", "[5,n]", "g_m of D_0..D_4"),
    "ptr_len": ("int32", "[n]", "pointer length L, 0 when P does not fire"),
    "ptr_cand": ("int32", "[n]", "pointer candidate token, -1 when none"),
    "ptr_pos": ("int64", "[n]", "pointer position in the group stream, -1 when none"),
    "bucket": ("int16", "[n]", "bucket id of the full component set (520 buckets) with the core base's k*"),
    "core_p_gold": ("float64", "[n]", "CVTM-core P(gold) (core base: see core_base)"),
    "core_top_ids": ("uint16", "[n,64]", "top-64 classes of the CVTM-core full distribution"),
    "core_top_p": ("float64", "[n,64]", "their probabilities"),
}


# ------------------------------------------------------------------ writing
def write_array(group_dir: Path, name: str, arr: np.ndarray) -> dict:
    """Save one array as <group_dir>/<name>.npy (dtype of ARRAYS enforced); returns its record."""
    want = np.dtype(ARRAYS[name][0])
    a = np.ascontiguousarray(arr)
    if a.dtype != want:
        raise TypeError(f"{name}: expected {want}, got {a.dtype}")
    path = Path(group_dir) / f"{name}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, a)
    return {"shape": list(a.shape), "dtype": str(a.dtype), "sha256": hashing.sha256_file(path),
            "array_sha256": hashing.sha256_array(a)}


def open_output(group_dir: Path, name: str, shape: tuple[int, ...]) -> np.memmap:
    """A writable .npy memmap for arrays filled in parallel (finalised by `finish_output`)."""
    path = Path(group_dir) / f"{name}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", dtype=np.dtype(ARRAYS[name][0]), shape=shape)


def finish_output(group_dir: Path, name: str) -> dict:
    path = Path(group_dir) / f"{name}.npy"
    a = np.load(path, mmap_mode="r")
    return {"shape": list(a.shape), "dtype": str(a.dtype), "sha256": hashing.sha256_file(path)}


# ------------------------------------------------------------------ parallel full-distribution pass
_STATE: dict[str, Any] = {}


def _dense_worker(args: tuple[int, int]) -> dict:
    lo, hi = args
    st = _STATE
    core, rec, gdir, k = st["core"], st["rec"], st["group_dir"], st["k"]
    names = ("kn_top_ids", "kn_top_p", "core_top_ids", "core_top_p")
    mm = {n: np.load(Path(gdir) / f"{n}.npy", mmap_mode="r+") for n in names}
    positions = np.arange(lo, hi, dtype=np.int64)

    def sink(a, b, pos, bi, bp, ci, cp):
        mm["kn_top_ids"][pos] = bi.astype(np.uint16)
        mm["kn_top_p"][pos] = bp
        mm["core_top_ids"][pos] = ci.astype(np.uint16)
        mm["core_top_p"][pos] = cp

    res = core.dense_pass(rec, k, positions, want_base=True, want_core=True, chunk=st["chunk"], sink=sink)
    for m in mm.values():
        m.flush()
    del mm
    return res["checks"]


def parallel_dense(core, rec, group_dir: Path, k: int = TOPK, workers: int = 1, chunk: int = 2048,
                   log=None) -> dict:
    """Fill kn_top_* and core_top_* of `group_dir` for every position of `rec` with `workers` forked
    processes (per-row results do not depend on the split, so the arrays are identical for any
    number of workers). Returns the merged checks and the file records."""
    n = int(rec.tok.shape[0])
    for name in ("kn_top_ids", "kn_top_p", "core_top_ids", "core_top_p"):
        m = open_output(group_dir, name, (n, k))
        del m
    # contiguous ranges of about 8 chunks each, handed out in order
    step = chunk * 8
    ranges = [(lo, min(n, lo + step)) for lo in range(0, n, step)]
    _STATE.update(core=core, rec=rec, group_dir=str(group_dir), k=int(k), chunk=int(chunk))
    t0 = time.perf_counter()
    checks: list[dict] = []
    try:
        if workers <= 1:
            for i, r in enumerate(ranges):
                checks.append(_dense_worker(r))
                if log and i % 25 == 0:
                    log(f"  dense pass {r[1]}/{n}")
        else:
            ctx = mp.get_context("fork")
            with ctx.Pool(int(workers)) as pool:
                for i, c in enumerate(pool.imap(_dense_worker, ranges)):
                    checks.append(c)
                    if log and i % 25 == 0:
                        log(f"  dense pass {ranges[i][1]}/{n}")
    finally:
        _STATE.clear()
    merged = {key: max(c[key] for c in checks) for key in checks[0] if key not in ("positions", "wall_s")}
    merged["positions"] = int(sum(c["positions"] for c in checks))
    merged["wall_s"] = time.perf_counter() - t0
    merged["workers"] = int(workers)
    files = {name: finish_output(group_dir, name) for name in ("kn_top_ids", "kn_top_p", "core_top_ids", "core_top_p")}
    return {"checks": merged, "files": files}


# ------------------------------------------------------------------ reading
def read_meta(cache_dir: str | Path) -> dict:
    return json.loads((Path(cache_dir) / META_FILE).read_text())


def load_arrays(cache_dir: str | Path, group: str, names: Iterable[str] | None = None,
                mmap: bool = True) -> dict[str, np.ndarray]:
    """{name: array} of one group ("hprime" or "validation"); memory-mapped read-only by default."""
    if group not in GROUPS:
        raise ValueError(f"group must be one of {GROUPS}")
    d = Path(cache_dir) / group
    names = list(ARRAYS) if names is None else list(names)
    out = {}
    for name in names:
        if name not in ARRAYS:
            raise KeyError(name)
        out[name] = np.load(d / f"{name}.npy", mmap_mode="r" if mmap else None)
    return out


def doc_features_of(arrays: dict[str, np.ndarray]) -> DocFeatures:
    """The cached document features of a group as a `DocFeatures` (whole-document window, pointer 5..64),
    for `tmlm.cvtm.mixture` (gold_components, bucket_ids)."""
    return DocFeatures(orders=D_ORDERS, window=None, ptr_min=5, ptr_max=64, n=np.asarray(arrays["doc_n"]),
                       g=np.asarray(arrays["doc_g"]), ptr_len=np.asarray(arrays["ptr_len"]),
                       ptr_cand=np.asarray(arrays["ptr_cand"]), ptr_pos=np.asarray(arrays["ptr_pos"]), n_fallback=0)


def role_mask(arrays: dict[str, np.ndarray], roles: str | int | Sequence[str | int]) -> np.ndarray:
    """bool [n]: positions of hprime documents whose role (name or residue) is in `roles`."""
    from .protocol import ROLE_RESIDUE

    if isinstance(roles, (str, int)):
        roles = [roles]
    res = [ROLE_RESIDUE[r] if isinstance(r, str) else int(r) for r in roles]
    per_pos = np.repeat(np.asarray(arrays["residue"]), np.diff(np.asarray(arrays["doc_start"])))
    return np.isin(per_pos, np.asarray(res))


def load_kn_variant(cache_dir: str | Path, variant: str, verify: bool = True):
    """The KNModel of one variant ("all", "noH", "noHp"); array hashes checked by `model.load_kn`."""
    from .model import load_kn

    if variant not in KN_VARIANTS:
        raise ValueError(f"variant must be one of {sorted(KN_VARIANTS)}")
    kn, _ = load_kn(Path(cache_dir) / KN_VARIANTS[variant], verify=verify)
    return kn


def load_core(cache_dir: str | Path, backend: str = "auto"):
    """The frozen CVTM-core (`CVTMCore.load`, state hash checked against cache_meta.json too)."""
    from .model import CVTMCore

    core = CVTMCore.load(Path(cache_dir) / CORE_FILE, backend=backend)
    want = read_meta(cache_dir)["core"]["state_hash"]
    if core.state_hash() != want:
        raise ValueError(f"core state hash {core.state_hash()} differs from cache_meta.json {want}")
    return core


def verify(cache_dir: str | Path) -> list[str]:
    """Problems found when re-hashing every file listed in cache_meta.json (empty: intact)."""
    cache_dir = Path(cache_dir)
    meta = read_meta(cache_dir)
    problems = []
    for rel, rec in meta["files"].items():
        p = cache_dir / rel
        if not p.exists():
            problems.append(f"{rel}: missing")
        elif hashing.sha256_file(p) != rec["sha256"]:
            problems.append(f"{rel}: sha256 differs")
    return problems


def file_records(cache_dir: str | Path, rels: Iterable[str]) -> dict[str, dict]:
    cache_dir = Path(cache_dir)
    return {rel: {"sha256": hashing.sha256_file(cache_dir / rel), "bytes": int((cache_dir / rel).stat().st_size)}
            for rel in rels}


def write_meta(cache_dir: str | Path, meta: dict) -> Path:
    path = Path(cache_dir) / META_FILE
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"format": FORMAT, **meta}, indent=1, sort_keys=True, default=str))
    os.replace(tmp, path)
    return path
