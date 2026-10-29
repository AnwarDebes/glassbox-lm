"""Rows as dicts of arrays: storage (bits packed), concatenation and subsets. numpy only (also used by the torch env).

A rows dict holds row arrays (one entry per position and candidate: pos, cand, y, cand_p, keys, bits, ...), the
position arrays named in POS_KEYS (one entry per position), pos_off (the row offsets of each position) and n_pos.
"""

from __future__ import annotations

import numpy as np

POS_KEYS = ("p_true", "top1", "ent", "pos_reply", "pos_t", "pos_dlg", "pos_x")


def expand_csr(starts: np.ndarray, lens: np.ndarray) -> np.ndarray:
    """Concatenation of arange(starts[i], starts[i] + lens[i]) for every i, without a Python loop."""
    starts, lens = np.asarray(starts, np.int64), np.asarray(lens, np.int64)
    tot = int(lens.sum())
    if tot == 0:
        return np.zeros(0, np.int64)
    first = np.repeat(np.cumsum(lens) - lens, lens)
    return np.repeat(starts, lens) + (np.arange(tot) - first)


def concat_rows(parts: list) -> dict:
    if len(parts) == 1:
        return parts[0]
    out = {}
    n_pos = np.cumsum([0] + [p["n_pos"] for p in parts])
    n_rows = np.cumsum([0] + [len(p["y"]) for p in parts])
    for k in parts[0]:
        if k == "n_pos":
            out[k] = int(n_pos[-1])
        elif k == "pos":
            out[k] = np.concatenate([p[k] + n_pos[i] for i, p in enumerate(parts)])
        elif k == "pos_off":
            out[k] = np.concatenate([[0]] + [p[k][1:] + n_rows[i] for i, p in enumerate(parts)])
        else:
            out[k] = np.concatenate([p[k] for p in parts])
    return out


def save_rows(path, rows: dict) -> None:
    d = {k: v for k, v in rows.items() if k != "bits"}
    if "bits" in rows:
        d["bits_packed"] = np.packbits(rows["bits"], axis=1)
        d["n_bits"] = rows["bits"].shape[1]
    np.savez_compressed(path, **d)


def load_rows(path, bits: bool = True) -> dict:
    z = np.load(path)
    out = {k: z[k] for k in z.files if k not in ("bits_packed", "n_bits")}
    out["n_pos"] = int(out["n_pos"])
    if bits and "bits_packed" in z.files:
        out["bits"] = np.unpackbits(z["bits_packed"], axis=1)[:, : int(z["n_bits"])]
    return out


def subset_rows(rows: dict, positions: np.ndarray) -> dict:
    """The rows of the given positions (in that order), re-indexed as a rows dict of their own."""
    positions = np.asarray(positions, np.int64)
    off = rows["pos_off"]
    ri = expand_csr(off[positions], off[positions + 1] - off[positions])
    sizes = off[positions + 1] - off[positions]
    out = {}
    n_rows = len(rows["y"])
    for k, v in rows.items():
        if k == "n_pos":
            out[k] = len(positions)
        elif k == "pos":
            out[k] = np.repeat(np.arange(len(positions)), sizes)
        elif k == "pos_off":
            out[k] = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        elif k in POS_KEYS:
            out[k] = v[positions]
        elif isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == n_rows:
            out[k] = v[ri]
        else:
            out[k] = v
    return out
