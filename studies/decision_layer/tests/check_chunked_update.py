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
