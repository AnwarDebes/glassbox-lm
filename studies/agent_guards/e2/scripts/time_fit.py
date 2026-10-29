"""Time one TM fit of the registered full grid's largest size (800 clauses, T 800, s 5, 30 epochs) on the largest
training set (held-out banking: slack + travel + workspace rows), to size the CPU budget. Decision-layer venv."""
import sys
import time
from pathlib import Path

E2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(E2 / "src"))
sys.path.insert(1, str(E2 / "scripts"))
import torch
import train_loso as TL

from e2guard.bits import QueryBits

torch.set_num_threads(2)
rows = TL.load_rows(sys.argv[1:])
qb = QueryBits(E2 / "data" / "query_bits" / "fold_banking.json")
fit = [r for r in rows if r["suite"] != "banking" and not r["_veto"]]
X, y, _ = TL.matrix(fit, qb)
for clauses in (200, 800):
    t0 = time.time()
    TL.fit_tm(X, y, clauses, 5.0, 1)
    print(f"clauses {clauses}: {time.time() - t0:.1f} s for {len(X)} rows x {X.shape[1]} bits, {TL.EPOCHS} epochs", flush=True)
