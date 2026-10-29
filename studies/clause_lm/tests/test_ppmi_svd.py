"""src/pair_baselines.py ppmi_svd(): the same input gives the same vectors (seeded SVD start vector, 2026-10-25).

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
        .venv-cdllm/bin/python tests/test_ppmi_svd.py [PAIR_BASELINES]

Run from studies/clause_lm. CPU only, a few seconds. PAIR_BASELINES defaults to src/pair_baselines.py; give another
copy (for example the version before 2026-10-25) to test it the same way. A random count matrix of 300 x 400 stands in
for the context counts; nothing is read from data/ and nothing is written.

Asserts:
- two calls on the same matrix give identical vectors (before 2026-10-25, scipy's svds drew its start vector from
  operating-system entropy, so they differed);
- the rows have unit length;
- another seed spans the same space: the cosine matrices of seed 0 and seed 1 agree to 1e-6, so a seed changes only
  rounding and signs, which k-means can still turn into other classes.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import toyutil  # noqa: E402

check = toyutil.Checks()


def main():
    target = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else toyutil.SRC / "pair_baselines.py"
    print(f"pair_baselines under test: {target}")
    mod, _ = toyutil.load_module_copy(target, "pair_baselines_under_test")
    rng = np.random.default_rng(4)
    M = rng.poisson(rng.gamma(0.3, 2.0, size=(300, 400)))
    E1, E2 = mod.ppmi_svd(M), mod.ppmi_svd(M)
    check(E1.shape == (300, 100), f"shape {E1.shape}")
    check(np.array_equal(E1, E2), f"two calls give identical vectors (max difference {np.abs(E1 - E2).max():.3g})")
    norms = np.linalg.norm(E1, axis=1)
    check(bool(np.all((np.abs(norms - 1) < 1e-9) | (norms < 1e-9))), "rows have unit length")
    try:
        E3 = mod.ppmi_svd(M, seed=1)
    except TypeError:
        check(False, "ppmi_svd() takes a seed")
        check.done()
    d = np.abs(E1 @ E1.T - E3 @ E3.T).max()
    check(d < 1e-6, f"seed 1 spans the same space as seed 0 (cosines differ by at most {d:.3g})")
    check.done()


if __name__ == "__main__":
    main()
