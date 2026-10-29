# The chunked update must match the unchunked one in distribution; check statistics on a small
# problem by forcing chunking (tiny CHUNK_ELEMS), and check that it trains to the same accuracy.
import sys; sys.path.insert(0, "src")
import numpy as np, torch
import dl.tm as T
from dl.tm import CoTM, TMConfig
rng = np.random.default_rng(0)
y = rng.integers(0, 7, 4000)
X = (rng.random((4000, 300)) < 0.2).astype(np.uint8); X[np.arange(4000), y] = 1; X[np.arange(4000), 100 + y] = 0
accs = {}
for chunk in (10**12, 20000):
    T.CHUNK_ELEMS = chunk
    a = []
    for seed in range(4):
        tm = CoTM(7, 300, TMConfig(n_clauses=200, T=200, s=5.0, epochs=8, batch_size=64, seed=seed)).fit(X[:3000], y[:3000])
        a.append(float((tm.predict(torch.as_tensor(X[3000:], device="cuda")).cpu().numpy() == y[3000:]).mean()))
    accs[chunk] = (np.mean(a), tm.stats()["mean_literals"])
print(accs)
# Bars added on 2026-10-25, before the first run of this file since the publication edits (it printed statistics and
# could not fail). They follow the comment above and the recorded output of 2026-10-16 (unchunked: accuracy 0.563, mean
# literals 2.655; chunked: 0.571, 2.25): both accuracies far above chance (1/7), the two within 0.03 of each other (mean
# of 4 seeds), and the mean clause lengths within 30% of each other.
(acc_u, lit_u), (acc_c, lit_c) = accs[10**12], accs[20000]
assert acc_u > 0.45 and acc_c > 0.45, accs
assert abs(acc_c - acc_u) < 0.03, accs
assert abs(lit_c - lit_u) <= 0.3 * max(lit_c, lit_u), accs
print("ok: chunked and unchunked updates train alike")
