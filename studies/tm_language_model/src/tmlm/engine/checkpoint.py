"""Checkpoint and resume for any EngineBase implementation (docs/04 M4: "checkpoint and resume so
long runs yield to timing runs"; docs/13 H2.2 replay).

A checkpoint is one `.npz` file holding the full canonical state (`STATE_FIELDS`), the
`EngineConfig` as JSON, and the RNG position. Because every draw of docs/11 section 3 is a pure
function of (seed, stream, example uid, clause, slot) and the epoch permutation is a pure
function of (seed, epoch, index), the "RNG position" of a training run is exactly the pair
(next epoch, next batch index) together with the `step` counter that already lives in the state.
There is no generator state to save: resuming at (epoch, batch_index) from the loaded state
reproduces the uninterrupted run bit for bit (tests/test_engine_checkpoint.py), on the CPU
engine and on the CUDA engine alike, and across the two.

    save_checkpoint(path, engine, epoch=e, batch_index=b)      # b = the NEXT batch to run
    cfg, state, pos = load_checkpoint(path)
    engine.load_state(state); resume from (pos["epoch"], pos["batch_index"])
    train_to(engine, epoch, batch_index, n_steps, checkpoint_path=..., every=...)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .api import STATE_FIELDS, EngineBase, state_hash_of
from .config import EngineConfig

FORMAT = "tmlm-engine-checkpoint-1"


def save_checkpoint(path: str | Path, engine: EngineBase, epoch: int, batch_index: int,
                    extra: dict[str, Any] | None = None, compress: bool = False) -> str:
    """Write the checkpoint; returns the state hash it holds. `batch_index` is the next batch to run.
    `compress=True` uses np.savez_compressed (a 1M-clause state is mostly LIT_EMPTY and CLS_EMPTY slots)."""
    state = engine.state()
    meta = {
        "format": FORMAT,
        "config": engine.cfg.to_dict(),
        "position": {"epoch": int(epoch), "batch_index": int(batch_index)},
        "step": int(state["step"][0]),
        "state_hash": state_hash_of(state, engine.cfg),
        "engine": type(engine).__name__,
        "extra": extra or {},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = np.savez_compressed if compress else np.savez
    writer(path, __meta__=np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8), **state)
    return meta["state_hash"]


def load_checkpoint(path: str | Path) -> tuple[EngineConfig, dict[str, np.ndarray], dict[str, Any]]:
    """Returns (config, state, meta). The stored state hash is verified against the arrays."""
    with np.load(Path(path)) as z:
        meta = json.loads(bytes(z["__meta__"]).decode("utf-8"))
        if meta.get("format") != FORMAT:
            raise ValueError(f"not a {FORMAT} file")
        state = {name: np.array(z[name]) for name, _, _ in STATE_FIELDS}
    cfg = EngineConfig(**meta["config"])
    if state_hash_of(state, cfg) != meta["state_hash"]:
        raise ValueError("checkpoint state hash does not match its arrays (file damaged)")
    return cfg, state, meta


def next_position(engine: EngineBase, epoch: int, batch_index: int) -> tuple[int, int]:
    """Position after running batch (epoch, batch_index)."""
    return (epoch + 1, 0) if batch_index + 1 >= engine.n_batches() else (epoch, batch_index + 1)


def train_to(engine: EngineBase, epoch: int, batch_index: int, n_steps: int,
             checkpoint_path: str | Path | None = None, every: int = 0,
             on_step: Any = None) -> tuple[int, int, dict[str, int]]:
    """Run `n_steps` batches from position (epoch, batch_index) in epoch order; optionally write a
    checkpoint every `every` steps (and at the end). Returns the next position and summed DIAG."""
    totals: dict[str, int] = {}
    e, b = epoch, batch_index
    for k in range(n_steps):
        d = engine.train_batch(e, b)
        for name, v in d.items():
            totals[name] = totals.get(name, 0) + v
        e, b = next_position(engine, e, b)
        if on_step is not None:
            on_step(k, e, b, d)
        if checkpoint_path is not None and every > 0 and ((k + 1) % every == 0 or k + 1 == n_steps):
            save_checkpoint(checkpoint_path, engine, e, b)
    return e, b, totals
