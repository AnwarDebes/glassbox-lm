"""Shared sampling primitives of the decoders (docs/11 section 3: counter-based RNG only).

Every random decision of a decoder is `rng32(seed, S_SAMPLER, a, b, c)` with
`a` = decoding step, `b` = window position, `c` = purpose (DRAW_TOKEN, ORDER_KEY, ELBO_RATE,
ELBO_MASK, REMASK). No Python `random`, no numpy global RNG anywhere on the path to the output.

Token draw: from a log-probability row, `p = exp(lp)`, the Q32 CDF `cdf[c] = floor(2^32 *
cumsum(p)[c])` with the last entry forced to 2^32, and the token is the smallest `c` with
`u < cdf[c]` for the 32-bit draw `u`. `checker_standalone.py` re-derives the same token from
the same numerators with the same float64 operations in the same order (bit-identical on one
platform and numpy build; the integer parts of a receipt do not depend on that).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..engine.config import Stream
from ..engine.rng import rng32

DRAW_TOKEN, ORDER_KEY, ELBO_RATE, ELBO_MASK, REMASK = 0, 1, 2, 3, 4
TWO32 = 1 << 32


@dataclass(frozen=True)
class Draw:
    """One recorded random decision (goes into the receipt)."""

    seed: int
    stream: int
    a: int
    b: int
    c: int
    u32: int


def draw(seed: int, a: int, b: int, c: int) -> Draw:
    return Draw(int(seed), int(Stream.SAMPLER), int(a), int(b), int(c), rng32(seed, int(Stream.SAMPLER), a, b, c))


def probs_from_numerators(num: np.ndarray, temperature: float, allowed: np.ndarray | None = None) -> np.ndarray:
    """float64 [V] probabilities from exact int64 logit numerators (logit = num / 2^24), softmax at
    `temperature`, optionally restricted to `allowed` (bool [V], renormalised). This exact sequence
    of float64 operations is what the standalone checker repeats."""
    z = num.astype(np.float64) / (float(1 << 24) * float(temperature))
    if allowed is not None:
        z = np.where(allowed, z, -np.inf)
    z = z - z.max()
    p = np.exp(z)
    p = p / p.sum()
    return p


def cdf_q32(p: np.ndarray) -> np.ndarray:
    cdf = np.floor(np.cumsum(p) * float(TWO32)).astype(np.int64)
    cdf[-1] = TWO32
    return cdf


def select(cdf: np.ndarray, u32: int) -> int:
    """Smallest c with u32 < cdf[c]."""
    return int(np.searchsorted(cdf, int(u32), side="right"))


def sample_token(num: np.ndarray, temperature: float, d: Draw, allowed: np.ndarray | None = None) -> int:
    return select(cdf_q32(probs_from_numerators(num, temperature, allowed)), d.u32)


def order_keys(seed: int, positions: np.ndarray, step: int = 0) -> np.ndarray:
    """A deterministic random key per position (for random any-order decoding)."""
    return np.array([rng32(seed, int(Stream.SAMPLER), step, int(p), ORDER_KEY) for p in positions], dtype=np.uint64)
