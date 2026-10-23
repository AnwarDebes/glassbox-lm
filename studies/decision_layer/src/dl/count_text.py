"""Text to bits with no neural network at any stage: the C4 word table and existsT (added 2026-10-19 for e72).

Plan: logs/plan_e72_count_table.md. A new file, so that no file hashed by the e70 and e71 freezes changes
(src/dl/static_text.py, dl/abcd.py, scripts/abcd_explore.py and abcd_confirm.py stay byte for byte as frozen).

- The table: data/tables/c4words_S1000_v1/table.npz, a copy of the frozen export c4words_S1000_v1 of the
  text-understanding line (project notes, not published). 63,091 C4 words x 500
  float16 rows, made by counting + SVD (PPMI-SVD) from 1B tokens of C4 en; nothing from potion or any neural model.
  The file's SHA-256 is checked at load against the constant and the MANIFEST; read with numpy only
  (allow_pickle=False); rows cast to float32 and L2-normalised.
- The tokenizer: a port of the export's tokenize_ref.py. Lower-case; "n't" and "'s 're 've 'll 'd 'm" split off at a
  word end; tokens = matches of n't|'[a-z]+|[a-z0-9]+|[^\\sa-z0-9]; a token not in the word list has no vector and is
  dropped (no subwords, no [UNK] row). An empty text is replaced by "(nothing yet)", as in static_text.
- ``ExistsTC``: dl.static_text.ExistsT with this table in place of potion's: the same fit (CPU spherical k-means,
  K 512, 25 iterations, a generator seeded 0, one thread; Thermo(8) on the train texts' max cosines), names,
  transform (lookup + OR) and transform_max. All table access goes through ``_table()``; ``info`` names the table.
- ``patch(name)``: for this process, dl.static_text.ExistsT builds ExistsTC on that table (so ABCDView's xcust,
  xrecent and xcrecent read it), and dl.static_text.table and StaticTable raise, so potion cannot be loaded.
  Used by scripts/table_run.py.
"""

from __future__ import annotations

import functools
import hashlib
import re
from pathlib import Path

import numpy as np
import torch

from dl import static_text
from dl.static_text import EMPTY, ExistsT, _sha256, one_thread

ROOT = Path(__file__).resolve().parents[2]
TABLES = {"c4words-S1000-v1": ("c4words_S1000_v1", "e1771d9eaf4a07f71c532fe4f3ef91909334cbdcb4d7785ff0bb7ed0e51c5c0a")}
TOKENIZE_REF_SHA256 = "2c3c1aff44ed34e017276221dff31ea29e7f9ab1f8f3a77cd0f193ee36e91803"
CONTR = [(re.compile(r"n't\b"), " n't"), (re.compile(r"'(s|re|ve|ll|d|m)\b"), r" '\1")]
TOK = re.compile(r"n't|'[a-z]+|[a-z0-9]+|[^\sa-z0-9]")


def tokens(text: str) -> list[str]:
    """The export's rule (tokenize_ref.tokens), unchanged."""
    t = text.lower()
    for pat, rep in CONTR:
        t = pat.sub(rep, t)
    return TOK.findall(t)


class CountTable:
    def __init__(self, name: str):
        sub, want = TABLES[name]
        d = ROOT / "data" / "tables" / sub
        man = {}
        for line in (d / "MANIFEST.sha256").read_text().splitlines():
            h, f = line.split()
            man[f] = h
        got = _sha256(d / "table.npz")
        if got != want or man.get("table.npz") != want:
            raise RuntimeError(f"{d / 'table.npz'}: sha256 {got}, constant {want}, MANIFEST {man.get('table.npz')}")
        if _sha256(d / "tokenize_ref.py") != TOKENIZE_REF_SHA256:
            raise RuntimeError(f"{d / 'tokenize_ref.py'}: not the reference this port follows")
        z = np.load(d / "table.npz", allow_pickle=False)
        self.words = [str(w) for w in z["words"]]
        self.row = {w: i for i, w in enumerate(self.words)}
        if len(self.row) != len(self.words):
            raise RuntimeError("duplicate words in the table")
        self.E = z["codes"].astype(np.float32)  # [V, 500]
        self.En = self.E / np.maximum(np.linalg.norm(self.E, axis=1, keepdims=True), 1e-12)
        self.table_sha256 = hashlib.sha256(np.ascontiguousarray(self.E).tobytes()).hexdigest()
        self.file_sha256 = got
        self.name = name
        self.tok = self  # ExistsT names concepts through tb.tok.id_to_token

    def id_to_token(self, i: int) -> str:
        return self.words[i]

    def ids(self, texts: list[str]) -> list[list[int]]:
        r = self.row
        return [[r[w] for w in tokens(t or EMPTY) if w in r] for t in texts]

    def meta(self) -> dict:
        return {"table": self.name, "table_file_sha256": self.file_sha256, "table_sha256": self.table_sha256,
                "tokenizer": f"port of tokenize_ref.py {TOKENIZE_REF_SHA256[:16]} (words; a token not in the list is dropped)",
                "n_words": len(self.words), "dim": int(self.E.shape[1])}


@functools.lru_cache(maxsize=4)
def count_table(name: str) -> CountTable:
    return CountTable(name)


class ExistsTC(ExistsT):
    """ExistsT on a count table: "some word of the <field> is close to concept k at level j", lookup + OR."""

    def __init__(self, K: int = 512, n_bins: int = 8, iters: int = 25, seed: int = 0, what: str = "text",
                 table_name: str = "c4words-S1000-v1", table_obj=None):
        super().__init__(K=K, n_bins=n_bins, iters=iters, seed=seed, what=what)
        self.table_name = table_name
        self._tb = table_obj  # tests only: another table object with ids, En, tok.id_to_token, table_sha256

    def _table(self):
        return self._tb if self._tb is not None else count_table(self.table_name)

    def _meta(self, tb) -> dict:
        if hasattr(tb, "meta"):
            return tb.meta()
        return {"table_sha256": tb.table_sha256, "hf_id": static_text.HF_ID, "revision": static_text.REV}

    # ExistsT.fit, transform and transform_max with table() replaced by self._table(); nothing else differs
    def fit(self, texts: list[str]) -> "ExistsTC":
        from dl.boolean import Thermo, _spherical_kmeans

        tb = self._table()
        ids = tb.ids(texts)
        vocab = np.array(sorted({t for x in ids for t in x}), dtype=np.int64)
        with one_thread():
            Z = torch.as_tensor(tb.En[vocab])
            gen = torch.Generator()
            gen.manual_seed(self.seed)
            C = _spherical_kmeans(Z, self.K, self.iters, gen)
            S_all = (torch.as_tensor(tb.En) @ C.T).numpy()  # [V, K]: every word of the table
            near = (Z @ C.T).T.topk(min(4, len(vocab)), dim=1).indices.numpy()
        self.C = C.numpy()
        mx = np.stack([S_all[x].max(0) if x else np.zeros(self.K, np.float32) for x in ids])
        self.th = Thermo(self.n_bins).fit(mx).th
        codes = np.concatenate([(S_all[:, k : k + 1] >= t[None, :]) for k, t in enumerate(self.th)], 1)
        self.n_bits = codes.shape[1]
        self.packed = np.packbits(codes, axis=1)  # [V, ceil(n_bits / 8)]
        self._S_all = S_all
        words = ["|".join(tb.tok.id_to_token(int(vocab[j])) for j in row) for row in near]
        self.names = [f"some word of the {self.what} is close to concept {k} ({words[k]}) at level {j} (cos >= {t:.3f})"
                      for k, th in enumerate(self.th) for j, t in enumerate(th)]
        th_flat = np.concatenate(self.th).astype(np.float32)
        self.info = {"what": self.what, "K": self.K, "n_bins": self.n_bins, "iters": self.iters, "seed": self.seed,
                     "draw": "cpu", "n_train_texts": len(texts), "n_train_vocab": int(len(vocab)), "n_bits": int(self.n_bits),
                     "concepts_sha256": hashlib.sha256(np.ascontiguousarray(self.C, dtype=np.float32).tobytes()).hexdigest(),
                     "thresholds_sha256": hashlib.sha256(th_flat.tobytes()).hexdigest(),
                     "codes_sha256": hashlib.sha256(self.packed.tobytes()).hexdigest(),
                     **self._meta(tb)}
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        """Lookup and OR: the bit code of every known word of the text, OR-ed (the run-time path)."""
        ids = self._table().ids(texts)
        out = np.zeros((len(ids), self.packed.shape[1]), np.uint8)
        for i, x in enumerate(ids):
            if x:
                out[i] = np.bitwise_or.reduce(self.packed[x], axis=0)
        return np.unpackbits(out, axis=1)[:, : self.n_bits]

    def transform_max(self, texts: list[str]) -> np.ndarray:
        """Reference: the max cosine to each concept over the text's known words, then the thresholds."""
        ids = self._table().ids(texts)
        mx = np.stack([self._S_all[x].max(0) if x else np.zeros(self.K, np.float32) for x in ids])
        return np.concatenate([(mx[:, k : k + 1] >= t[None, :]) for k, t in enumerate(self.th)], 1).astype(np.uint8)


def _no_potion(*_a, **_k):
    raise RuntimeError("dl.count_text.patch is active: potion's table must not be loaded in this run")


def patch(name: str) -> dict:
    """For this process: ABCDView's existsT parts read the named count table; potion's table cannot be loaded."""
    tb = count_table(name)  # load and check now, before any view is built
    static_text.table = _no_potion
    static_text.StaticTable = _no_potion
    static_text.ExistsT = functools.partial(ExistsTC, table_name=name)
    return tb.meta()
