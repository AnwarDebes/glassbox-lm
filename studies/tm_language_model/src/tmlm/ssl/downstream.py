"""Downstream text tasks of docs/13 amendment A7 (Q_text): AG News, SST-2 and TREC-6, TRAIN SPLITS ONLY.

Sources (the pinned revisions of the early transfer test, experiments/ssl_transfer_early.py): the train
parquet file of each dataset in the local Hugging Face cache `data/hf_cache/hub`. A missing train file is
downloaded (that file only) when the caller allows it; nothing else is ever downloaded or opened.

Hard refusal (A7): no test split of any dataset, and not SST-2's validation split
(read by the exploratory runs of 2026-09-24, so A7 bars it from every confirmatory number), is ever
opened: `open_split` and every path helper accept exactly the registered train file and raise
`ForbiddenSplit` for anything else. tests/test_ssl_r1_downstream.py proves it with an audit hook on file
opens while loading all three tasks.

Carving by the docs/11 section 3 mix64 of the example index i (the row index in the train file):
    r(i) = mix64(i) mod 20
    confirm  r = 0, SST-2 only: the A7 confirmatory SST-2 set (5 percent of SST-2 train). It is excluded from
             every training and label sample, and this module never returns its texts.
    dev      r in {1, 2}: the exploration dev set ("carved from each task's train split by a fixed hash
             (10 percent)", A7). The same hash as the confirm rule, so dev and confirm are disjoint by
             construction.
    pool     everything else (AG News and TREC: including r = 0): the label pool.

Data hashes: A7 asks for write-once files docs/notes/preregistration-data-hashes-A7-<dataset>.json.
`write_hash_file` creates one with an exclusive create (never overwrites; an existing file is re-read and
verified). Until it exists, `verify_train_file` checks the train parquet against the A6 hash file of the same
revision (read only).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..common import hashing
from ..common.runlog import project_root
from .pretext import mix64_np

HF_CACHE_REL = Path("data") / "hf_cache" / "hub"
CARVE_MOD = 20
CONFIRM_RESIDUES = (0,)
DEV_RESIDUES = (1, 2)

TASKS: dict[str, dict] = {
    "sst2": {"code": 1, "repo": "nyu-mll/glue", "revision": "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c",
             "train_file": "sst2/train-00000-of-00001.parquet", "text": ("sentence",), "label": "label",
             "n_classes": 2, "has_confirm_carve": True,
             "licence": "unknown (A7: SST-2 card 'unknown'; GLUE card: refer to the original licences)"},
    "agnews": {"code": 2, "repo": "fancyzhx/ag_news", "revision": "eb185aade064a813bc0b7f42de02595523103ca4",
               "train_file": "data/train-00000-of-00001.parquet", "text": ("text",), "label": "label",
               "n_classes": 4, "has_confirm_carve": False, "licence": "non-commercial research (A7)"},
    "trec": {"code": 3, "repo": "CogComp/trec", "revision": "65752bf53af25bc935a0dce92fb5b6c930728450",
             "train_file": "default/train/0000.parquet", "text": ("text",), "label": "coarse_label",
             "n_classes": 6, "has_confirm_carve": False, "licence": "unknown (A7: TREC card 'unknown')"},
}


class ForbiddenSplit(PermissionError):
    """Raised for any attempt to open a split other than the registered train file."""


def _root(root: Path | None) -> Path:
    return Path(root) if root is not None else project_root()


def hash_file_path(task: str, amendment: str = "A7", root: Path | None = None) -> Path:
    return _root(root) / "docs" / "notes" / f"preregistration-data-hashes-{amendment}-{task}.json"


def _task(task: str) -> dict:
    if task not in TASKS:
        raise KeyError(f"unknown task {task!r}")
    return TASKS[task]


def check_allowed(task: str, rel: str) -> None:
    """Refuse every file of the dataset repo except the registered train file."""
    t = _task(task)
    if rel != t["train_file"]:
        raise ForbiddenSplit(f"{task}: refusing to open {rel!r}; only the train file {t['train_file']!r} may be "
                             "opened (docs/13 A7: no test split, and not SST-2 validation)")


def open_split(task: str, split: str, root: Path | None = None, allow_download: bool = False) -> Path:
    """The local path of a split's file. Only split == 'train' is allowed; every other split (test,
    validation, dev, ...) raises ForbiddenSplit before any file is touched."""
    if split != "train":
        raise ForbiddenSplit(f"{task}: split {split!r} is refused; R1 reads train splits only (docs/13 A7)")
    return train_file(task, root=root, allow_download=allow_download)


def local_path(task: str, rel: str, root: Path | None = None) -> Path:
    check_allowed(task, rel)
    t = _task(task)
    org, name = t["repo"].split("/")
    return _root(root) / HF_CACHE_REL / f"datasets--{org}--{name}" / "snapshots" / t["revision"] / rel


def train_file(task: str, root: Path | None = None, allow_download: bool = False) -> Path:
    """Path of the train parquet in the local cache; downloads that one file if it is missing and
    `allow_download` is set, else raises FileNotFoundError."""
    t = _task(task)
    p = local_path(task, t["train_file"], root)
    if p.exists():
        return p
    if not allow_download:
        raise FileNotFoundError(f"{p} is missing (pass allow_download=True to fetch the train file only)")
    from huggingface_hub import hf_hub_download

    check_allowed(task, t["train_file"])
    got = Path(hf_hub_download(t["repo"], t["train_file"], repo_type="dataset", revision=t["revision"],
                               cache_dir=str(_root(root) / HF_CACHE_REL)))
    return got


def read_train(task: str, root: Path | None = None, allow_download: bool = False) -> tuple[list[str], np.ndarray]:
    """(texts, labels int64) of the train file, in file order (row index = example index)."""
    import pyarrow.parquet as pq

    t = _task(task)
    path = train_file(task, root, allow_download)
    with open(path, "rb") as fh:  # a Python open, so audit hooks see every file this module reads (tests)
        tab = pq.read_table(fh, columns=list(t["text"]) + [t["label"]])
    cols = {c: tab.column(c).to_pylist() for c in t["text"]}
    n = tab.num_rows
    texts = [" ".join(str(cols[c][i]) for c in t["text"]) for i in range(n)]
    y = np.asarray(tab.column(t["label"]).to_pylist(), dtype=np.int64)
    if y.min() < 0 or y.max() >= t["n_classes"]:
        raise ValueError(f"{task}: label outside 0..{t['n_classes'] - 1}")
    return texts, y


# ------------------------------------------------------------------ carving
def residues(n: int) -> np.ndarray:
    """int64 [n]: mix64(i) mod 20 for the example indices i = 0..n-1."""
    return (mix64_np(np.arange(int(n), dtype=np.uint64)) % np.uint64(CARVE_MOD)).astype(np.int64)


def carve(task: str, n: int) -> dict[str, np.ndarray]:
    """Ascending int64 example indices of pool, dev and confirm (confirm empty unless SST-2)."""
    t = _task(task)
    r = residues(n)
    confirm = np.isin(r, CONFIRM_RESIDUES) if t["has_confirm_carve"] else np.zeros(n, bool)
    dev = np.isin(r, DEV_RESIDUES)
    pool = ~(confirm | dev)
    return {"pool": np.flatnonzero(pool).astype(np.int64), "dev": np.flatnonzero(dev).astype(np.int64),
            "confirm": np.flatnonzero(confirm).astype(np.int64)}


@dataclass
class TaskData:
    task: str
    n_train: int
    pool_text: list[str]
    pool_y: np.ndarray
    pool_idx: np.ndarray       # example indices into the train file
    dev_text: list[str]
    dev_y: np.ndarray
    dev_idx: np.ndarray
    n_confirm_excluded: int
    train_sha256: str

    @property
    def n_classes(self) -> int:
        return int(TASKS[self.task]["n_classes"])


def load_task(task: str, root: Path | None = None, allow_download: bool = False, verify: bool = True) -> TaskData:
    """Pool and dev of a task from its train file; the SST-2 confirm carve is dropped here and never returned."""
    texts, y = read_train(task, root, allow_download)
    sha = hashing.sha256_file(train_file(task, root))
    if verify:
        v = verify_train_file(task, root, sha=sha)
        if not v["equal"]:
            raise ValueError(f"{task}: train file sha256 {sha} differs from {v['reference_file']}")
    c = carve(task, len(texts))
    return TaskData(task=task, n_train=len(texts),
                    pool_text=[texts[i] for i in c["pool"]], pool_y=y[c["pool"]], pool_idx=c["pool"],
                    dev_text=[texts[i] for i in c["dev"]], dev_y=y[c["dev"]], dev_idx=c["dev"],
                    n_confirm_excluded=int(c["confirm"].shape[0]), train_sha256=sha)


# ------------------------------------------------------------------ hashes
def verify_train_file(task: str, root: Path | None = None, sha: str | None = None) -> dict:
    """Compare the train parquet's sha256 with the A7 hash file, or with the A6 file of the same revision
    while no A7 file exists (read only)."""
    t = _task(task)
    sha = sha or hashing.sha256_file(train_file(task, root))
    for amend in ("A7", "A6"):
        p = hash_file_path(task, amend, root)
        if p.exists():
            rec = json.loads(p.read_text(encoding="utf-8"))
            ref = rec["files"].get(t["train_file"], {}).get("sha256")
            same_rev = rec.get("source", {}).get("revision") == t["revision"]
            return {"sha256": sha, "reference_file": str(p.relative_to(_root(root))), "reference_sha256": ref,
                    "equal": bool(ref == sha and same_rev)}
    return {"sha256": sha, "reference_file": None, "reference_sha256": None, "equal": False}


def idx_sha(a: np.ndarray) -> str:
    return hashing.sha256_bytes(np.asarray(a, np.int64).astype("<i8").tobytes())


def hash_record(task: str, root: Path | None = None) -> dict:
    """The A7 hash-file record of a task (train file only; carve rule and carve hashes)."""
    t = _task(task)
    path = train_file(task, root)
    texts, y = read_train(task, root)
    c = carve(task, len(texts))
    return {
        "purpose": "docs/13 amendment A7 (family H17): downstream data hashed once before any R1 use; train split only",
        "written_by": "src/tmlm/ssl/downstream.py write_hash_file (experiments/ssl_r1_pretext.py --stage data); write-once, never edited",
        "source": {"repo": t["repo"], "repo_type": "dataset", "revision": t["revision"], "licence": t["licence"],
                   "access": "the train parquet only, from data/hf_cache/hub (downloaded with huggingface_hub.hf_hub_download only when missing), read with pyarrow"},
        "files": {t["train_file"]: {"sha256": hashing.sha256_file(path), "bytes": int(path.stat().st_size), "split": "train"}},
        "refused": "every test split; SST-2 validation (read by the exploratory runs of 2026-09-24, A7 Disclosure)",
        "carve": {"rule": "r(i) = mix64(i) mod 20 over the example index i (docs/11 section 3 mix64); confirm r = 0 (SST-2 only), dev r in {1, 2}, pool = the rest",
                  "n_train": len(texts),
                  "pool": {"n": int(c["pool"].shape[0]), "idx_sha256": idx_sha(c["pool"]),
                           "class_counts": np.bincount(y[c["pool"]], minlength=t["n_classes"]).tolist()},
                  "dev": {"n": int(c["dev"].shape[0]), "idx_sha256": idx_sha(c["dev"]),
                          "class_counts": np.bincount(y[c["dev"]], minlength=t["n_classes"]).tolist()},
                  "confirm": {"n": int(c["confirm"].shape[0]), "idx_sha256": idx_sha(c["confirm"]),
                              "note": "SST-2: the A7 confirmatory set, excluded from every training and label sample" if t["has_confirm_carve"] else "none for this task"},
                  "row_index_format": "int64 little endian, ascending"},
        "text_columns": list(t["text"]), "label_column": t["label"], "n_classes": t["n_classes"],
    }


def write_hash_file(task: str, root: Path | None = None) -> dict:
    """Create docs/notes/preregistration-data-hashes-A7-<task>.json once (exclusive create). An existing file
    is re-read and compared (sha256 of the train file and the carve hashes); returns {path, written, equal}."""
    rec = hash_record(task, root)
    path = hash_file_path(task, "A7", root)
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        rel = TASKS[task]["train_file"]
        same = (old["files"][rel]["sha256"] == rec["files"][rel]["sha256"]
                and all(old["carve"][k]["idx_sha256"] == rec["carve"][k]["idx_sha256"] for k in ("pool", "dev", "confirm")))
        return {"path": str(path), "written": False, "equal": bool(same)}
    rec["written_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(path, "x", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1, sort_keys=True)
        fh.write("\n")
    return {"path": str(path), "written": True, "equal": True}
