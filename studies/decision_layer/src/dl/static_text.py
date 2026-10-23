"""Text to bits with no neural network at run time: a static word table and bit codes per token.

The table is potion-base-32M (Model2Vec; distilled once, offline, from bge-base-en-v1.5): 63,091 tokens x
512 floats and a rule-based tokenizer. A port, not an import, of the recipe in the project notes (not published)
(speed_exists.py lines 27 to 61, exists_tm.py lines 46 to 102, graph_feats.py lines 42 to 70: the existsT of the
text-understanding line); nothing there is imported or edited. Added 2026-10-17 for the no-network agent
(the plan in the project notes, not published).

- ``table()``: the table and tokenizer, loaded once per process from the local cache at the pinned
  revision, with the file SHA-256s checked. ``ids`` tokenizes without special tokens (as the model's
  own StaticEmbedding does); ``mean`` is the normalised average of the token vectors, which equals
  SentenceTransformer.encode(..., normalize_embeddings=True) for this model (tests/test_static_text.py).
- ``ExistsT``: K concepts = spherical k-means (dl.boolean._spherical_kmeans, k-means++ seeding, 25
  iterations) of the distinct token ids of the training texts, on the CPU with a generator seeded 0
  and one thread: the same recipe as the text-understanding line's existsT with a CPU draw (that line drew on
  the GPU, so the concepts are not the same). sim_k(text) = max over the text's tokens of cos(token, concept k); bits
  "sim_k >= t" with Thermo(8) thresholds (percentiles 12.5 to 87.5, unique, per concept) fitted on the
  training texts. Every token of the whole table gets its precomputed bit code, so ``transform`` is a
  lookup and a bitwise OR per text (a word never seen in training still has a code); ``transform_max``
  computes the same bits from the max cosine, as a reference.
- Empty text: ``EMPTY`` ("(nothing yet)") stands in, in fitting and in use, as in the bge-large caches.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import json

import numpy as np
import torch

HF_ID = "minishlab/potion-base-32M"
REV = "1e5a03f8eeb2c98b928fbbd846f22f816360919f"
FILE_SHA256 = {
    "model.safetensors": "99f6c33204c9231a7391871b7a3c91409b532c8f587a9ea44fc282303d8dec28",
    "tokenizer.json": "7d75cbc54318138807c401b0f0c9721117c628b39de8e8e0edb6cb17e0ee7d18",
}
EMPTY = "(nothing yet)"


@contextlib.contextmanager
def one_thread():
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(n)


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class StaticTable:
    def __init__(self):
        from huggingface_hub import hf_hub_download
        from sentence_transformers import SentenceTransformer

        for name, want in FILE_SHA256.items():
            got = _sha256(hf_hub_download(HF_ID, name, revision=REV, local_files_only=True))
            if got != want:
                raise RuntimeError(f"{HF_ID}@{REV} {name}: sha256 {got} != {want}")
        self.st = SentenceTransformer(HF_ID, device="cpu", revision=REV)
        self.tok = self.st[0].tokenizer
        self.E = self.st[0].embedding.weight.detach().float().cpu().numpy()  # [V, 512]
        self.En = self.E / np.maximum(np.linalg.norm(self.E, axis=1, keepdims=True), 1e-12)
        self.table_sha256 = hashlib.sha256(np.ascontiguousarray(self.E).tobytes()).hexdigest()

    def ids(self, texts: list[str]) -> list[list[int]]:
        return [e.ids for e in self.tok.encode_batch([t or EMPTY for t in texts], add_special_tokens=False)]

    def mean(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.E.shape[1]), np.float32)
        for i, x in enumerate(self.ids(texts)):
            if x:
                m = self.E[x].mean(0)
                out[i] = m / max(float(np.linalg.norm(m)), 1e-12)
        return out


@functools.lru_cache(maxsize=1)
def table() -> StaticTable:
    return StaticTable()


class ExistsT:
    """existsT bits of one text field: "some word of the <field> is close to concept k at level j"."""

    def __init__(self, K: int = 512, n_bins: int = 8, iters: int = 25, seed: int = 0, what: str = "text"):
        self.K, self.n_bins, self.iters, self.seed, self.what = K, n_bins, iters, seed, what

    def fit(self, texts: list[str]) -> "ExistsT":
        from dl.boolean import Thermo, _spherical_kmeans

        tb = table()
        ids = tb.ids(texts)
        vocab = np.array(sorted({t for x in ids for t in x}), dtype=np.int64)
        with one_thread():
            Z = torch.as_tensor(tb.En[vocab])
            gen = torch.Generator()
            gen.manual_seed(self.seed)
            C = _spherical_kmeans(Z, self.K, self.iters, gen)
            S_all = (torch.as_tensor(tb.En) @ C.T).numpy()  # [V, K]: every token of the table
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
                     "table_sha256": tb.table_sha256, "hf_id": HF_ID, "revision": REV}
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        """Lookup and OR: the bit code of every token of the text, OR-ed (the run-time path)."""
        ids = table().ids(texts)
        out = np.zeros((len(ids), self.packed.shape[1]), np.uint8)
        for i, x in enumerate(ids):
            if x:
                out[i] = np.bitwise_or.reduce(self.packed[x], axis=0)
        return np.unpackbits(out, axis=1)[:, : self.n_bits]

    def transform_max(self, texts: list[str]) -> np.ndarray:
        """Reference: the max cosine to each concept over the text's tokens, then the thresholds."""
        ids = table().ids(texts)
        mx = np.stack([self._S_all[x].max(0) if x else np.zeros(self.K, np.float32) for x in ids])
        return np.concatenate([(mx[:, k : k + 1] >= t[None, :]) for k, t in enumerate(self.th)], 1).astype(np.uint8)

    def info_json(self) -> str:
        return json.dumps(self.info, sort_keys=True)
