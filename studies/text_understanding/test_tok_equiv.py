import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
# tok_tm2.py (memory-lean products) must give bit-identical states and scores to tok_tm_v1.py (the code of the
# first phase A rows) for the same seed: 3 training steps on HWU64 S+P+N and on banking77 G+S, then val scores
import importlib.util, sys
import numpy as np, torch
S = f"{_REPO_ROOT}/studies/text_understanding/emb_potion_test2"
def mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
v1 = mod(f"{_REPO_ROOT}/studies/text_understanding/tok_tm_v1.py", "v1")
v2 = mod(f"{_REPO_ROOT}/studies/text_understanding/tok_tm.py", "v2")  # was tok_tm2.py when run (16:29 UTC)
from dl.tm import TMConfig
for ds, parts in (("hwu64", "S+P+N"), ("banking77", "G+S")):
    z = np.load(f"{S}/{ds}__graph_K512.npz")
    F = v1.Feats(z, parts, ("train", "val"), "cuda")
    C = int(z["y_train"].max()) + 1
    cfg = TMConfig(n_clauses=4000, T=4000, s=8.0, seed=1)
    a, b = v1.TokTM(C, F.f, cfg), v2.TokTM(C, F.f, cfg)
    y = torch.as_tensor(z["y_train"], device="cuda")
    for k in range(3):
        idx = torch.arange(128 * k, 128 * (k + 1), device="cuda")
        a.step(F, idx, y[idx]); b.step(F, idx, y[idx])
    same_state = torch.equal(a.ta, b.ta) and torch.equal(a.w, b.w)
    sa, sb = a.scores(F, "val"), b.scores(F, "val")
    print(ds, parts, "state identical:", same_state, "val scores identical:", torch.equal(sa, sb),
          "included literals:", int((a.ta > 256).sum()))
