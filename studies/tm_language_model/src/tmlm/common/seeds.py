"""Deterministic seeding (protocol rule 4).

ORDER MATTERS FOR GraphTM. `GraphTsetlinMachine/tm.py` creates its curand generator at
IMPORT time with `g = curandom.XORWOWRandomNumberGenerator()` (tm.py line 36 of the
vendored copy) [verified in source]. With no `seed_getter`, pycuda seeds that generator
from the GLOBAL legacy numpy RNG: `np.random.randint(0, (1 << 31) - 1, generator_count)`
(pycuda 2025.1.3, curandom.py lines 756 to 761) [verified in source]. Therefore:

    from tmlm.common.seeds import seed_everything
    rng = seed_everything(seed)             # FIRST
    from GraphTsetlinMachine.tm import ...  # only now

Seeding after that import cannot change the curand stream any more, so
`seed_everything` refuses to run late unless `allow_late=True` is passed. GraphTM
results also depend on the launch configuration, so a run must log (seed, grid, block)
together.

Other packages, for contrast:
  - PyHierarchicalTsetlinMachineCUDA (git c85ca4e) builds its curand generator in the
    model constructor, not at import: pass `seed=` to the constructor. With `seed=None`
    it draws from the global np.random at construction time (tm.py lines 106 to 110)
    [verified in source], so seed_everything must still come before construction.
  - tmu's CPU TMAutoEncoder path uses unseeded C `rand()` [stated: design notes]: such runs
    are recorded as non-reproducible, no seed can fix them.
  - The Track B engine uses its own counter based RNG keyed by the seed in its config.
  - PYTHONHASHSEED only takes effect at interpreter start. env/env.sh exports it.
"""

from __future__ import annotations

import random
import sys

import numpy as np

# Modules that fix a GPU random stream from np.random at the moment they are imported.
IMPORT_TIME_SEEDED_MODULES = ("GraphTsetlinMachine.tm",)

MAX_SEED = 2**32 - 1


def late_seeded_modules() -> list[str]:
    """Names from IMPORT_TIME_SEEDED_MODULES that are already imported."""
    return [name for name in IMPORT_TIME_SEEDED_MODULES if name in sys.modules]


def seed_everything(seed: int, *, allow_late: bool = False) -> np.random.Generator:
    """Seed Python `random` and the global numpy RNG, and return a fresh Generator.

    Args:
        seed: int in [0, 2**32 - 1] (the range `np.random.seed` accepts). bool is
            rejected.
        allow_late: pass True to seed even though a module of
            IMPORT_TIME_SEEDED_MODULES is already imported. Its curand stream is then
            NOT controlled by this call.

    Returns:
        `np.random.default_rng(seed)`: a PCG64 Generator that is independent of the
        global legacy state. New code should draw from this Generator and leave the
        global RNG to the libraries that insist on it.

    Raises:
        TypeError, ValueError: bad seed.
        RuntimeError: called after an import-time seeded module was imported.
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    seed = int(seed)
    if not 0 <= seed <= MAX_SEED:
        raise ValueError(f"seed must be in [0, {MAX_SEED}], got {seed}")
    late = late_seeded_modules()
    if late and not allow_late:
        raise RuntimeError(
            "seed_everything was called after importing "
            + ", ".join(late)
            + ": the curand generator of that module was already seeded from np.random "
            "at import time. Seed first, import second (or pass allow_late=True)."
        )
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)
