"""Reference implementation of the deterministic randomness of docs/11-engine-spec.md section 3.

Pure Python integers, no numpy, so that there is exactly one obvious reading of the spec.
The C++ / CUDA core (`core/tm_core.h`) re-implements these functions independently and is
checked against golden vectors generated from this file.
"""

from __future__ import annotations

M64 = (1 << 64) - 1
_STREAM_MUL = 0xD6E8FEB86659FD93


def mix64(z: int) -> int:
    """SplitMix64 finalizer, all arithmetic modulo 2^64."""
    z = (z + 0x9E3779B97F4A7C15) & M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return z ^ (z >> 31)


def rng64(seed: int, stream: int, a: int, b: int, c: int) -> int:
    """Counter-based draw keyed by (seed, stream, a, b, c). Every argument is taken mod 2^64."""
    z = mix64((seed ^ ((stream * _STREAM_MUL) & M64)) & M64)
    z = mix64((z + a) & M64)
    z = mix64((z + b) & M64)
    return mix64((z + c) & M64)


def rng32(seed: int, stream: int, a: int, b: int, c: int) -> int:
    return rng64(seed, stream, a, b, c) >> 32


def bern(p_q32: int, seed: int, stream: int, a: int, b: int, c: int) -> bool:
    """Bernoulli draw with success probability p_q32 / 2^32."""
    return rng32(seed, stream, a, b, c) < p_q32


def pick(n: int, seed: int, stream: int, a: int, b: int, c: int) -> int:
    """Uniform-ish integer in [0, n), n < 2^32, by multiply-shift."""
    return (rng32(seed, stream, a, b, c) * n) >> 32


def q32(p: float) -> int:
    """Host-side conversion of a probability to a Q32 threshold: min(2^32 - 1, floor(p * 2^32))."""
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"probability out of range: {p}")
    return min((1 << 32) - 1, int(p * 4294967296.0))


def inv_q32(s: float | int) -> int:
    """Q32 threshold of 1/s. Exact integer division when s is an integer."""
    if s < 1:
        raise ValueError("s must be >= 1")
    if float(s).is_integer():
        return min((1 << 32) - 1, (1 << 32) // int(s))
    return q32(1.0 / float(s))


def perm_bits(n: int) -> int:
    """Smallest EVEN bit width b with 2^b >= n (at least 2)."""
    b = 2
    while (1 << b) < n:
        b += 2
    return b


def perm(seed: int, stream_perm: int, epoch: int, x: int, n: int) -> int:
    """Epoch permutation of [0, n): 4-round Feistel with cycle walking (spec section 3)."""
    if not 0 <= x < n:
        raise ValueError("x out of range")
    h = perm_bits(n) // 2
    m = (1 << h) - 1
    while True:
        left, right = x >> h, x & m
        for r in range(4):
            left, right = right, left ^ (rng32(seed, stream_perm, epoch, r, right) & m)
        x = (left << h) | right
        if x < n:
            return x
