"""The network-free test on the newtable path (plan logs/plan_e82_refined_table_agent.md; added 2026-10-21).

    HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-dl/bin/python tests/test_newtable_path.py

CPU only; ABCD train and dev only. In a fresh interpreter, the path a registered run takes is followed:
- scripts/newtable_run.register for data/tables/c4words_a3B_v1;
- dl.count_text.patch on that name, as table_run.py does;
- ABCDView with view B's parts (mem, words, xcust, xrecent), fitted on ABCD train with the empty "none" cache and
  transformed on 300 dev rows;
- a CPU TM decision through dl.cpu_tm_fast, with a random clause bank of the view's width.
It asserts that:
- sentence_transformers, model2vec and transformers were never imported;
- potion's table cannot load;
- the only encoder cache opened is data/emb/abcd__none.npz, and the only table opened is c4words_a3B_v1;
- the bits' text blocks were built on the a3B table (static_info names its file hash).
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLE_SHA = "4a1ed2aed7aa4ce726892446a2cdf34b23a6e43d3c2e943f0fb46c5210d7d855"

CHILD = r'''
import builtins, sys
from pathlib import Path
ROOT = Path(sys.argv[1]); TABLE_SHA = sys.argv[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(1, str(ROOT / "scripts"))
opened = []
_open = builtins.open
def spy(file, *a, **k):
    opened.append(str(file))
    return _open(file, *a, **k)
builtins.open = spy
import numpy as np
from newtable_run import register
from dl import count_text, cpu_tm_fast, static_text
register(Path("data/tables/c4words_a3B_v1"), "c4words-a3B-v1")
count_text.patch("c4words-a3B-v1")
from dl.abcd import ABCDView, load_examples
tr, bl = load_examples("train", "abcd")
dv, _ = load_examples("dev", "abcd")
E = np.load(ROOT / "data" / "emb" / "abcd__none.npz")
emb = {s: {f: E[f"{s}__{f}"] for f in ("cust", "recent", "full")} for s in ("train", "dev")}
v = ABCDView(parts=("mem", "words", "xcust", "xrecent"), lex_k=1000, seed=0).fit(tr, emb["train"], bl)
X = v.transform(dv[:300], {f: a[:300] for f, a in emb["dev"].items()})
rng = np.random.default_rng(0)
F = X.shape[1]
ta = rng.integers(0, 512, size=(64, 2 * F))
ta[:, :] = np.where(rng.random((64, 2 * F)) < 0.002, 300, 100)  # a few included literals per clause
w = rng.integers(-5, 6, size=(64, len(bl)))
pred = cpu_tm_fast.class_sums(cpu_tm_fast.pack(ta, w), X).argmax(1)
assert pred.shape == (300,)
bad = [m for m in ("sentence_transformers", "model2vec", "transformers") if m in sys.modules]
assert not bad, f"imported: {bad}"
try:
    static_text.table()
    raise SystemExit("potion's table loaded")
except RuntimeError:
    pass
emb_files = sorted({o for o in opened if "/data/emb/" in o})
assert emb_files == [str(ROOT / "data" / "emb" / "abcd__none.npz")], emb_files
tables = sorted({Path(o).parent.name for o in opened if "/data/tables/" in o})
assert tables == ["c4words_a3B_v1"], tables
info = v.static_info()
assert all(info[k]["table_file_sha256"] == TABLE_SHA for k in ("xcust", "xrecent")), info
print(f"ok: view B {F} bits on the a3B table; CPU decisions on 300 dev rows; no network module imported; "
      f"encoder caches opened {emb_files}; tables opened {tables}")
'''


def test_newtable_path_is_network_free():
    r = subprocess.run([sys.executable, "-c", CHILD, str(ROOT), TABLE_SHA], cwd=ROOT, capture_output=True, text=True,
                       env={"HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": "",
                            "PATH": "/usr/bin:/bin"})
    print(r.stdout.strip())
    assert r.returncode == 0, r.stderr[-3000:]


if __name__ == "__main__":
    test_newtable_path_is_network_free()
    print("all checks passed")
