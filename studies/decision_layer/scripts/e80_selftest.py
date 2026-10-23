# e80 self-test (CPU, dev bits only): view B + xcrecent keeps view B's bits as a prefix, byte for byte
import hashlib, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dl import count_text
count_text.patch("c4words-S1000-v1")
from dl.abcd import ABCDView, load_examples
REF = {"star": "e74c_star_Bc4_c16000_s16", "abcd": "e74c_abcd_Bc4_c4000_s16", "multiwoz": "e77_mwoz_Bc4_c16000_s8"}
rows = {d["tag"]: d for d in map(json.loads, (ROOT / "results/explore/abcd.jsonl").read_text().splitlines()) if d.get("arm") == "tm" and d.get("tag") in REF.values()}
for ds, tag in REF.items():
    tr, bl = load_examples("train", ds); dv, _ = load_examples("dev", ds)
    E = np.load(ROOT / "data/emb" / f"{ds}__none.npz"); emb = {s: {f: E[f"{s}__{f}"] for f in ("cust", "recent", "full")} for s in ("train", "dev")}
    v = ABCDView(parts=("mem", "words", "xcust", "xrecent", "xcrecent"), lex_k=1000, seed=0).fit(tr, emb["train"], bl)
    X = v.transform(dv, emb["dev"])
    F0 = len(ABCDView(parts=("mem", "words", "xcust", "xrecent"), lex_k=1000, seed=0).fit(tr, emb["train"], bl).names)
    h = hashlib.sha256(np.ascontiguousarray(X[:, :F0], dtype=np.uint8).tobytes()).hexdigest()
    ok = h == rows[tag]["bits_sha256"]["dev"]
    print(f"{'ok  ' if ok else 'FAIL'} {ds}: B + xcrecent = {X.shape[1]} bits; the first {F0} equal {tag}'s dev bits ({h[:8]}); xcrecent {X.shape[1] - F0} bits; on-share {X[:, F0:].mean():.3f}", flush=True)
    if not ok: sys.exit(1)
print("all checks passed")
