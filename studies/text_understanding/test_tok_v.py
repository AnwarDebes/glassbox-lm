import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
# V part = bag of training words of the sentence, the same row for every token
import importlib.util, numpy as np, torch
spec = importlib.util.spec_from_file_location("t3", f"{_REPO_ROOT}/studies/text_understanding/tok_tm.py"); t3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(t3)
z = np.load(f"{_REPO_ROOT}/studies/text_understanding/emb_potion_test2/hwu64__graph_K512.npz")
F = t3.Feats(z, "V+W", ("train", "val"), "cuda")
wv = list(z["train_vocab"]); pos = {t: i for i, t in enumerate(wv)}
tok, off = z["val__tok"], z["val__off"]
ok = True
idx = torch.arange(0, 300, device="cuda")
X, mask = F.batch("val", idx)
for i in range(300):
    want = sorted({pos[t] for t in tok[off[i]:off[i + 1]] if t in pos})
    Vrow = X[i, 0, : F.n_w].nonzero().flatten().tolist()
    Wrows = sorted({j for l in range(int(mask[i].sum())) for j in X[i, l, F.n_w:].nonzero().flatten().tolist()})
    ok &= Vrow == want and Wrows == want and all(torch.equal(X[i, 0, : F.n_w], X[i, l, : F.n_w]) for l in range(X.shape[1]))
print("V = bag of words and W per token consistent on 300 val sentences:", ok, "f =", F.f)
