"""Confirmatory R1 of docs/13 amendment A7 (family H17) with the additions of amendment A8 and the readings of
amendment A9 (binding): the library of experiments/ssl_r1_confirm.py (pure functions and small classes; no run
logic, no GPU requirement except where an array module `xp` is passed in).

Contents
  1. Confirmatory evaluation sets, counted (A7 "Splits", A9 item 10): AG News and TREC-6 official TEST splits and
     the SST-2 confirm carve (mix64(i) mod 20 == 0 of SST-2 train), every load recorded through the test-access
     counter of tmlm.data.evalsets BEFORE any byte of the set is read, at most once per (frozen configuration, task)
     unless a re-access reason is given (logged). The MEN test pairs of Q_con go through the same guard. The sha256
     of each test file is taken BEFORE the first access from the host's published file metadata (no content) into a
     write-once reference file; every access verifies the downloaded bytes against it. Per-process access lines
     (the log's pid) let a stage prove that it made no access.
  2. Frozen probes: the R1 probes (L2 logistic regression with C by 4-fold CV inside the labelled sample, the tmu
     Coalesced TM probe) refactored so that the refit model is RETURNED (tmlm.ssl.transfer and tmlm.ssl.probes only
     return accuracies); the same code paths, tested equal. Multinomial naive Bayes and NBSVM (Wang and Manning 2012)
     for A8 item 1.
  3. Statistics of A8 item 1 (A9 item 5): the two-level bootstrap (seeds, then units within each resampled seed;
     10,000 resamples) and the seed-level Student-t interval.
  4. The registered verdict rules of H17.1 to H17.4, H17.11, H17.12 and the A8 "literal-learning credit" label, as
     pure functions of their inputs; estimate rows without a threshold (A9 item 5); the winner rule (A9 item 1:
     fixed A3, the seeds 1 to 3 re-selection rule is computed and printed, never used).
  5. Q_int: NPMI coherence and an automated word-intrusion score of clause concept groups.
  6. The A8 random conjunction bank (1 to 3 position-bound literals drawn with the pretraining text's literal
     frequencies, no training) and its featurisation (numpy or cupy; a pure-Python reference for tests).
  7. Q_con: the A7 word code (fired-clause indicators summed over up to 1,000 occurrences of the word with all its
     subtokens hidden, PPMI-weighted, cosine), all clauses and dimension-matched columns.

Readings: docs/13 amendment A9 fixes the readings that decide verdict inputs; the rest are in
experiments/cvtm/configs/ssl_r1_confirm.json ("readings") and in the docstrings below; every one of them was
written before any confirmatory output existed.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from ..common import hashing
from ..common.runlog import project_root
from . import downstream as D
from . import transfer as TR
from .pretext import mix64_np

# ====================================================================== 1. confirmatory sets (counted)
CONFIRM_SOURCES: dict[str, dict] = {
    "agnews": {"kind": "test_split", "file": "data/test-00000-of-00001.parquet"},
    "trec": {"kind": "test_split", "file": "default/test/0000.parquet"},
    "sst2": {"kind": "confirm_carve"},
}
ACCESS_TAG = "A7 R1 confirmatory evaluation set"
MEN_TAG = "A7 H17.3 Q_con MEN test pairs"
ACCESS_CALLER = "ssl.confirm.record_confirm_access"


class ConfirmAccessRefused(PermissionError):
    """A second access of the same (configuration, task) without a logged re-access reason."""


def _root(root: Path | None) -> Path:
    return Path(root) if root is not None else project_root()


def _data_root(root: Path | None) -> Path | None:
    """The data root of the test-access log: None (tmlm.data.corpus_store.data_root, which honours $TMLM_DATA_DIR)
    unless a project root is given (then <root>/data)."""
    return None if root is None else Path(root) / "data"


def access_key(config_id: str, task: str) -> str:
    return f"config={config_id}; task={task};"


def prior_accesses(config_id: str, task: str, root: Path | None = None) -> list[dict]:
    """Lines of the test-access log that already recorded (config_id, task)."""
    from ..data import evalsets as EV

    path = EV.access_log_path(_data_root(root))
    if not path.exists():
        return []
    key = access_key(config_id, task)
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                if key in str(rec.get("purpose", "")):
                    out.append(rec)
    return out


def record_confirm_access(tag: str, config_id: str, task: str, detail: str, root: Path | None = None,
                          reaccess_reason: str | None = None) -> dict:
    """Count one access of the confirmatory set of `task` for `config_id` (tmlm.data.evalsets.record_test_access).
    Refuses a second access of the same pair unless `reaccess_reason` is given (then written into the purpose)."""
    from ..data import evalsets as EV

    prior = prior_accesses(config_id, task, root)
    if prior and not (reaccess_reason and reaccess_reason.strip()):
        raise ConfirmAccessRefused(f"{access_key(config_id, task)} was already accessed {len(prior)} time(s) "
                                   f"(first {prior[0].get('utc')}); a re-access needs a logged reason")
    purpose = f"{tag}; {access_key(config_id, task)} {detail}"
    if prior:
        purpose += f"; REACCESS {len(prior) + 1}: {reaccess_reason.strip()}"
    before = EV.test_access_count(_data_root(root))
    after = EV.record_test_access(ACCESS_CALLER, purpose, root=_data_root(root))
    return {"before": int(before), "after": int(after), "purpose": purpose, "prior_accesses": len(prior)}


def own_accesses(root: Path | None = None, pid: int | None = None) -> list[dict]:
    """Lines of the test-access log written by process `pid` (default: this process). The log records the pid of
    every access (tmlm.data.evalsets.record_test_access), so a stage can prove that IT made no access even while
    other parallel jobs make counted accesses (docs/13 A9 item 11; the chain's shared lock is the second guard)."""
    from ..data import evalsets as EV

    path = EV.access_log_path(_data_root(root))
    if not path.exists():
        return []
    me = os.getpid() if pid is None else int(pid)
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                if int(rec.get("pid", -1)) == me:
                    out.append(rec)
    return out


class ConfirmPreflightFailed(RuntimeError):
    """Something a counted access needs is missing or wrong; raised BEFORE any access is recorded."""


def test_file_path(task: str, root: Path | None = None) -> Path:
    t = D.TASKS[task]
    src = CONFIRM_SOURCES[task]
    if src["kind"] != "test_split":
        raise KeyError(f"{task} has no confirmatory test split file")
    org, name = t["repo"].split("/")
    return _root(root) / D.HF_CACHE_REL / f"datasets--{org}--{name}" / "snapshots" / t["revision"] / src["file"]


def fetch_test_file(task: str, root: Path | None = None, allow_download: bool = True) -> Path:
    """The local test parquet (downloaded with huggingface_hub, that file only, when missing and allowed)."""
    p = test_file_path(task, root)
    if p.exists():
        return p
    if not allow_download:
        raise FileNotFoundError(f"{p} is missing")
    from huggingface_hub import hf_hub_download

    t = D.TASKS[task]
    return Path(hf_hub_download(t["repo"], CONFIRM_SOURCES[task]["file"], repo_type="dataset", revision=t["revision"],
                                cache_dir=str(_root(root) / D.HF_CACHE_REL)))


# ---------------------------------------------------------------------- test hash references (docs/13 A9 item 10)
TEST_HASH_SOURCE = ("the dataset host's published file metadata at the pinned revision (huggingface_hub HfApi.get_paths_info), "
                    "taken before the first counted access; no content downloaded")
# A file the host stores in git rather than LFS has no published sha256, only its git blob id (SHA-1 over
# b"blob <size>\\0" + content). A9 item 10 names sha256, so such a file is refused unless the config names an amendment
# that allows the git blob id as the reference ("git_blob_sha1").
NON_LFS_MODES = ("refuse", "git_blob_sha1")


def git_blob_sha1(path: Path, chunk: int = 1 << 20) -> str:
    """The git blob id of a file: sha1(b'blob <size>\\0' + content), what the host publishes for a non-LFS file."""
    import hashlib

    size = int(Path(path).stat().st_size)
    h = hashlib.sha1(f"blob {size}\0".encode("ascii"))
    with open(path, "rb") as fh:
        while True:
            b = fh.read(int(chunk))
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def test_hash_path(task: str, root: Path | None = None) -> Path:
    """Write-once reference hash file of a confirmatory TEST file (A9 item 10). The A7 files
    preregistration-data-hashes-A7-<task>.json are write-once and hold the train files only."""
    return _root(root) / "docs" / "notes" / f"preregistration-data-hashes-A7-{task}-test.json"


def repo_file_metadata(info: Any) -> dict:
    """A huggingface_hub RepoFile (or a dict of the same fields) as plain JSON: path, size, git blob id, LFS sha256,
    size and pointer size, Xet hash."""
    get = (lambda k, d=None: info.get(k, d)) if isinstance(info, dict) else (lambda k, d=None: getattr(info, k, d))
    lfs = get("lfs")
    if lfs is not None and not isinstance(lfs, dict):
        lfs = {"sha256": getattr(lfs, "sha256", None), "size": getattr(lfs, "size", None),
               "pointer_size": getattr(lfs, "pointer_size", None)}
    return {"path": get("path"), "size": get("size"), "blob_id": get("blob_id"), "lfs": lfs, "xet_hash": get("xet_hash"),
            "type": type(info).__name__}


def test_file_metadata(task: str) -> dict:
    """Published metadata of the confirmatory TEST file of `task` at the pinned revision (HfApi.get_paths_info on
    exactly that path; metadata only, no content download). Raises ValueError when the host returns anything but one
    file entry for that path."""
    from huggingface_hub import HfApi

    t = D.TASKS[task]
    rel = CONFIRM_SOURCES[task]["file"]
    infos = HfApi().get_paths_info(t["repo"], [rel], revision=t["revision"], repo_type="dataset")
    got = [repo_file_metadata(i) for i in infos]
    if len(got) != 1 or got[0]["path"] != rel:
        raise ValueError(f"{task}: get_paths_info({t['repo']}@{t['revision']}, {rel!r}) returned {got}")
    return {"repo": t["repo"], "repo_type": "dataset", "revision": t["revision"], "call": "HfApi.get_paths_info", **got[0]}


def _test_hash_record(task: str, digest: str, size: int, meta: dict | None, kind: str = "sha256", amendment: str | None = None) -> dict:
    t = D.TASKS[task]
    rel = CONFIRM_SOURCES[task]["file"]
    return {"purpose": "docs/13 amendment A9 item 10 (family H17): content hash of a confirmatory TEST file, taken from the dataset "
                       "host's published file metadata at the pinned revision BEFORE the first counted access (no content "
                       "downloaded); every counted access verifies the downloaded bytes against it; write-once, never edited",
            "written_by": "src/tmlm/ssl/confirm.py write_test_hash_reference (experiments/ssl_r1_confirm.py --stage testhash)",
            "source": {"repo": t["repo"], "repo_type": "dataset", "revision": t["revision"], "licence": t["licence"],
                       "hash_source": TEST_HASH_SOURCE},
            "files": {rel: {"reference": kind, "sha256": str(digest) if kind == "sha256" else None,
                            "git_blob_sha1": str(digest) if kind == "git_blob_sha1" else None, "bytes": int(size), "split": "test",
                            "metadata": meta, "amendment": amendment}},
            "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def _read_json_retry(path: Path, tries: int = 50, wait_s: float = 0.1) -> dict:
    """json of a file another process may be writing right now (empty or partial for a moment): retried."""
    last: Exception | None = None
    for _ in range(int(tries)):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            last = exc
            time.sleep(wait_s)
    raise ValueError(f"{path}: unreadable after {tries} tries ({last})")


def test_hash_reference(task: str, root: Path | None = None) -> dict:
    """{'hash_file', 'reference' (sha256 or git_blob_sha1), 'digest', 'sha256' (None for a git blob reference),
    'bytes'} of the reference of a confirmatory TEST file. FileNotFoundError when the testhash stage has not written
    it; ValueError when it names another repo, revision or file."""
    hp = test_hash_path(task, root)
    if not hp.exists():
        raise FileNotFoundError(f"{hp} is missing: run the testhash stage first (docs/13 A9 item 10)")
    t = D.TASKS[task]
    rel = CONFIRM_SOURCES[task]["file"]
    rec = _read_json_retry(hp)
    src = rec.get("source", {})
    if src.get("repo") != t["repo"] or src.get("revision") != t["revision"] or rel not in rec.get("files", {}):
        raise ValueError(f"{hp} does not describe {t['repo']}@{t['revision']}:{rel}")
    f = rec["files"][rel]
    kind = str(f.get("reference", "sha256"))
    if kind not in ("sha256", "git_blob_sha1") or not f.get(kind):
        raise ValueError(f"{hp}: no usable reference ({kind})")
    return {"hash_file": str(hp), "reference": kind, "digest": str(f[kind]), "sha256": f.get("sha256"), "bytes": int(f["bytes"])}


def verify_or_write_test_hash(task: str, digest: str, size: int, root: Path | None = None, meta: dict | None = None,
                              kind: str = "sha256", amendment: str | None = None) -> dict:
    """Create the write-once reference with an exclusive create. When the file exists, including when another process
    created it between any check and our create (FileExistsError: the create is the only test), it is re-read and
    must hold the same kind of reference, digest and size; a difference raises ValueError."""
    hp = test_hash_path(task, root)
    hp.parent.mkdir(parents=True, exist_ok=True)
    rec = _test_hash_record(task, digest, size, meta, kind, amendment)
    out = {"hash_file": str(hp), "equal": True, "reference": kind, "digest": str(digest), "bytes": int(size),
           "sha256": str(digest) if kind == "sha256" else None}
    try:
        with open(hp, "x", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=1, sort_keys=True)
            fh.write("\n")
        return {**out, "written": True}
    except FileExistsError:
        pass
    ref = test_hash_reference(task, root)
    if ref["reference"] != kind or ref["digest"] != str(digest) or ref["bytes"] != int(size):
        raise ValueError(f"{task}: {kind} {digest} / {size} bytes differ from the reference {hp} "
                         f"({ref['reference']} {ref['digest']} / {ref['bytes']})")
    return {**out, "written": False}


def write_test_hash_reference(task: str, root: Path | None = None, metadata_fn: Callable[[str], dict] | None = None,
                              non_lfs: str = "refuse", amendment: str | None = None) -> dict:
    """The testhash stage for one task: the host's file metadata (metadata_fn, default test_file_metadata), then the
    write-once reference: the published LFS sha256 (A9 item 10). A file without an LFS sha256 (stored in git) raises
    ValueError unless non_lfs == 'git_blob_sha1' AND an amendment is named, then its published git blob id is the
    reference. No counted access, no content download."""
    if non_lfs not in NON_LFS_MODES:
        raise ValueError(f"non_lfs must be one of {NON_LFS_MODES}")
    meta = (metadata_fn or test_file_metadata)(task)
    lfs = meta.get("lfs") or {}
    if lfs.get("sha256") and lfs.get("size") is not None:
        if meta.get("size") is not None and int(meta["size"]) != int(lfs["size"]):
            raise ValueError(f"{task}: file size {meta['size']} differs from the LFS size {lfs['size']}")
        out = verify_or_write_test_hash(task, str(lfs["sha256"]), int(lfs["size"]), root, meta, "sha256")
        return {**out, "metadata": meta}
    blob = str(meta.get("blob_id") or "")
    if non_lfs != "git_blob_sha1":
        raise ValueError(f"{task}: the host publishes no LFS sha256 for {CONFIRM_SOURCES[task]['file']} (a file stored in git: "
                         f"blob id {blob or None}, {meta.get('size')} bytes); A9 item 10 names sha256, so no reference is written "
                         "(the git blob id can serve only under a named amendment: non_lfs='git_blob_sha1')")
    if not amendment:
        raise ValueError("the git blob id as the reference needs a named amendment")
    if len(blob) != 40 or any(ch not in "0123456789abcdef" for ch in blob) or meta.get("size") is None:
        raise ValueError(f"{task}: no usable git blob id in {meta}")
    out = verify_or_write_test_hash(task, blob, int(meta["size"]), root, meta, "git_blob_sha1", amendment)
    return {**out, "metadata": meta}


def verify_test_file(task: str, path: Path, root: Path | None = None) -> dict:
    """The downloaded bytes of a confirmatory TEST file against its reference (the sha256, or the git blob id where an
    amendment allowed it, and the size); ValueError on a difference. The sha256 of the bytes is always returned."""
    ref = test_hash_reference(task, root)
    sha = hashing.sha256_file(path)
    size = int(Path(path).stat().st_size)
    got = sha if ref["reference"] == "sha256" else git_blob_sha1(path)
    if got != ref["digest"] or size != ref["bytes"]:
        raise ValueError(f"{task}: downloaded test file {path} has {ref['reference']} {got} / {size} bytes; the reference "
                         f"{ref['hash_file']} holds {ref['digest']} / {ref['bytes']}")
    return {"hash_file": ref["hash_file"], "written": False, "equal": True, "reference": ref["reference"], "digest": got,
            "sha256": sha, "bytes": size}


def sst2_confirm_reference(root: Path | None = None) -> dict:
    """The references of the SST-2 confirm carve, checked without reading the carve: the A7 hash file holds the carve
    rows' idx_sha256 and the train file's sha256 equals it (the train file is hashed, not parsed)."""
    hp = D.hash_file_path("sst2", "A7", root)
    if not hp.exists():
        raise FileNotFoundError(f"{hp} is missing")
    ref = json.loads(hp.read_text(encoding="utf-8"))["carve"]["confirm"]["idx_sha256"]
    v = D.verify_train_file("sst2", root)
    if not v["equal"]:
        raise ValueError(f"sst2: train file sha256 {v['sha256']} differs from {v['reference_file']}")
    return {"hash_file": str(hp), "carve_idx_sha256": ref, "train_sha256": v["sha256"]}


def preflight_references(tasks: Sequence[str], root: Path | None = None) -> dict:
    """Every reference a counted access verifies against, checked BEFORE the access: the A9 item 10 test hash files
    of AG News and TREC, the SST-2 carve and train-file hashes. Raises ConfirmPreflightFailed."""
    out = {}
    for task in tasks:
        try:
            if CONFIRM_SOURCES[task]["kind"] == "test_split":
                out[task] = test_hash_reference(task, root)
            else:
                out[task] = sst2_confirm_reference(root)
        except (OSError, ValueError, KeyError) as exc:
            raise ConfirmPreflightFailed(f"{task}: {exc}") from exc
    return out


def men_test_reference(root: Path | None = None, rel: str = "data/concept_eval/men/MEN/MEN_dataset_lemma_form.test") -> dict:
    """The A6 hash-file entry of the MEN test pairs and the file's presence, checked before the counted access."""
    hp = _root(root) / "docs" / "notes" / "preregistration-data-hashes-A6-men.json"
    try:
        ref = json.loads(hp.read_text(encoding="utf-8"))["files"][rel]["sha256"]
    except (OSError, KeyError, ValueError) as exc:
        raise ConfirmPreflightFailed(f"MEN test reference: {exc}") from exc
    if not (_root(root) / rel).exists():
        raise ConfirmPreflightFailed(f"{rel} is missing")
    return {"hash_file": str(hp), "sha256": ref, "file": rel}


@dataclass
class ConfirmSet:
    task: str
    texts: list[str]
    y: np.ndarray
    info: dict


def _read_parquet(path: Path, text_cols: Sequence[str], label_col: str, n_classes: int, task: str) -> tuple[list[str], np.ndarray]:
    import pyarrow.parquet as pq

    with open(path, "rb") as fh:
        tab = pq.read_table(fh, columns=list(text_cols) + [label_col])
    cols = {c: tab.column(c).to_pylist() for c in text_cols}
    texts = [" ".join(str(cols[c][i]) for c in text_cols) for i in range(tab.num_rows)]
    y = np.asarray(tab.column(label_col).to_pylist(), dtype=np.int64)
    if y.size and (y.min() < 0 or y.max() >= n_classes):
        raise ValueError(f"{task}: label outside 0..{n_classes - 1}")
    return texts, y


def load_confirm(task: str, config_id: str, detail: str, root: Path | None = None, allow_download: bool = True,
                 reaccess_reason: str | None = None) -> ConfirmSet:
    """THE counted loader of the confirmatory set of `task` (one call = one counted access). The reference hash is
    checked to exist BEFORE the access (a missing reference refuses with no access); the access is recorded before
    the file is located, downloaded, hashed or read; the downloaded bytes are verified against the reference before
    they are parsed (docs/13 A9 item 10)."""
    src = CONFIRM_SOURCES[task]
    t = D.TASKS[task]
    if src["kind"] == "test_split":
        test_hash_reference(task, root)
    elif not D.hash_file_path(task, "A7", root).exists():
        raise FileNotFoundError(f"{D.hash_file_path(task, 'A7', root)} is missing")
    acc = record_confirm_access(ACCESS_TAG, config_id, task, detail, root, reaccess_reason)
    if src["kind"] == "test_split":
        path = fetch_test_file(task, root, allow_download)
        h = verify_test_file(task, path, root)
        texts, y = _read_parquet(path, t["text"], t["label"], t["n_classes"], task)
        info = {"source": "official test split", "file": src["file"], "rows": len(texts), **h}
    else:
        texts_all, y_all = D.read_train(task, root)
        v = D.verify_train_file(task, root)
        if not v["equal"]:
            raise ValueError(f"{task}: train file sha256 differs from {v['reference_file']}")
        c = D.carve(task, len(texts_all))
        rows = c["confirm"]
        ref = json.loads(D.hash_file_path(task, "A7", root).read_text(encoding="utf-8"))["carve"]["confirm"]["idx_sha256"]
        if D.idx_sha(rows) != ref:
            raise ValueError(f"{task}: confirm carve rows differ from the A7 hash file")
        texts, y = [texts_all[i] for i in rows.tolist()], y_all[rows]
        info = {"source": "SST-2 confirm carve: mix64(i) mod 20 == 0 of the train file (A7 Splits)",
                "train_sha256": v["sha256"], "carve_idx_sha256": ref, "rows": int(rows.shape[0])}
    info.update({"access": acc, "n": len(texts)})
    return ConfirmSet(task, texts, y, info)


def load_men_test(config_id: str, detail: str, root: Path | None = None, reaccess_reason: str | None = None,
                  rel: str = "data/concept_eval/men/MEN/MEN_dataset_lemma_form.test") -> tuple[list[tuple[str, str, float]], dict]:
    """The MEN test pairs (counted, once per configuration); the A6 reference is checked before the access, the
    file's sha256 must equal it."""
    from . import concepts as KC

    ref = men_test_reference(root, rel)["sha256"]
    acc = record_confirm_access(MEN_TAG, config_id, "men_test", detail, root, reaccess_reason)
    path = _root(root) / rel
    sha = hashing.sha256_file(path)
    if sha != ref:
        raise ValueError(f"{rel}: sha256 {sha} differs from the A6 hash file")
    return KC.parse_men_lemma(path), {"file": rel, "sha256": sha, "access": acc}


# ====================================================================== 2. frozen probes
@dataclass
class Frozen:
    """A fitted probe: kind in {lr, tm, mnb, nbsvm}, the refit model, the chosen hyperparameter and the CV table."""
    kind: str
    model: Any
    choice: Any
    cv: dict
    dense: bool = False
    extra: dict = field(default_factory=dict)


def fit_lr(Xl, yl, seed: int, dense: bool, c_grid: Sequence[float], folds: int, max_iter: int, backend: str = "cpu") -> Frozen:
    """tmlm.ssl.transfer.lr_cell (the R1 primary probe) with the refit model returned: identical folds, grid, tie
    rule (smaller C) and fitting code (TR._fit_lr)."""
    yl = np.asarray(yl)
    splits = TR.cv_splits(yl, seed, folds)
    cv = {}
    for C in c_grid:
        accs = []
        for tr, va in splits:
            m = TR._fit_lr(Xl[tr], yl[tr], C, dense, max_iter, backend)
            accs.append(float(np.mean(TR._predict_lr(m, Xl[va], dense) == yl[va])))
        cv[float(C)] = float(np.mean(accs))
    best = max(cv.values())
    C = min(c for c, a in cv.items() if a == best)
    return Frozen("lr", TR._fit_lr(Xl, yl, C, dense, max_iter, backend), C, {"folds": len(splits), "acc": cv}, dense)


def _tm_fit(Xl: np.ndarray, yl: np.ndarray, cfg: dict, seed: int, epochs: int):
    """tmlm.ssl.probes.tm_fit_predict without the prediction: the same seeding and construction."""
    from tmu.models.classification.coalesced_classifier import TMCoalescedClassifier

    yl = np.asarray(yl, np.int64)
    if np.unique(yl).shape[0] < 2:
        return ("const", int(yl[0]))
    np.random.seed(int(seed) % (2 ** 31))
    tm = TMCoalescedClassifier(number_of_clauses=int(cfg["clauses"]), T=int(cfg["T"]), s=float(cfg["s"]),
                               platform="CPU", seed=int(seed) % (2 ** 31))
    Xl = np.ascontiguousarray(Xl, dtype=np.uint32)
    ylu = np.ascontiguousarray(yl, dtype=np.uint32)
    for _ in range(int(epochs)):
        tm.fit(Xl, ylu)
    return ("tm", tm)


def tm_predict(model, X: np.ndarray) -> np.ndarray:
    if model[0] == "const":
        return np.full(X.shape[0], model[1], dtype=np.int64)
    return np.asarray(model[1].predict(np.ascontiguousarray(X, dtype=np.uint32))).astype(np.int64)


def fit_tm(Xl: np.ndarray, yl, seed: int, grid: Sequence[dict], epochs: int, folds: int) -> Frozen:
    """tmlm.ssl.probes.tm_probe with the refit model returned (CV through PR.tm_fit_predict itself)."""
    from . import probes as PR

    yl = np.asarray(yl, np.int64)
    splits = TR.cv_splits(yl, seed, folds)
    cv = {}
    for gi, cfg in enumerate(grid):
        accs = []
        for fi, (tr, va) in enumerate(splits):
            p = PR.tm_fit_predict(Xl[tr], yl[tr], Xl[va], cfg, seed * 1000 + fi, epochs)
            accs.append(float(np.mean(p == yl[va])))
        cv[gi] = float(np.mean(accs))
    best = max(cv.values())
    gi = min(g for g, a in cv.items() if a == best)
    return Frozen("tm", _tm_fit(Xl, yl, grid[gi], seed, epochs), dict(grid[gi]), {"folds": len(splits), "acc": cv})


class MNB:
    """Multinomial naive Bayes on count features with Lidstone smoothing alpha and the empirical class prior
    (sklearn MultinomialNB with fit_prior=True; tests check equal joint log-likelihoods and predictions)."""

    def __init__(self, alpha: float) -> None:
        self.alpha = float(alpha)

    def fit(self, X, y) -> "MNB":
        import scipy.sparse as sp

        y = np.asarray(y, np.int64)
        self.classes_ = np.unique(y)
        Y = (y[:, None] == self.classes_[None, :]).astype(np.float64)
        X = sp.csr_matrix(X, dtype=np.float64)
        fc = np.asarray((X.T @ Y).T)                      # [K, F]
        cc = Y.sum(axis=0)
        sfc = fc + self.alpha
        self.feature_log_prob_ = np.log(sfc) - np.log(sfc.sum(axis=1, keepdims=True))
        self.class_log_prior_ = np.log(cc) - np.log(cc.sum())
        return self

    def joint_log_likelihood(self, X) -> np.ndarray:
        import scipy.sparse as sp

        return np.asarray(sp.csr_matrix(X, dtype=np.float64) @ self.feature_log_prob_.T) + self.class_log_prior_

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self.joint_log_likelihood(X), axis=1)]


def nb_log_ratio(Xb, ybin: np.ndarray, alpha: float) -> np.ndarray:
    """Wang and Manning (2012) r = log((p / |p|_1) / (q / |q|_1)), p = alpha + sum of the positive rows, q = alpha +
    sum of the negative rows of the BINARISED features."""
    import scipy.sparse as sp

    Xb = sp.csr_matrix(Xb, dtype=np.float64)
    ybin = np.asarray(ybin, bool)
    p = alpha + np.asarray(Xb[ybin].sum(axis=0)).ravel()
    q = alpha + np.asarray(Xb[~ybin].sum(axis=0)).ravel()
    return np.log(p / p.sum()) - np.log(q / q.sum())


class NBSVM:
    """NBSVM of Wang and Manning (2012): binarised features scaled by the NB log-count ratio r, an L2-regularised
    L2-loss linear SVM (sklearn LinearSVC, dual coordinate descent, the LIBLINEAR solver of the paper) and the
    interpolation w' = (1 - beta) * mean(|w|) + beta * w (the bias kept). Multiclass: one-vs-rest, argmax of the
    decision values; two classes: one model."""

    def __init__(self, C: float, alpha: float = 1.0, beta: float = 0.25, seed: int = 0, max_iter: int = 10000) -> None:
        self.C, self.alpha, self.beta, self.seed, self.max_iter = float(C), float(alpha), float(beta), int(seed), int(max_iter)

    def _one(self, Xb, ybin: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        import warnings

        import scipy.sparse as sp
        from sklearn.svm import LinearSVC

        r = nb_log_ratio(Xb, ybin, self.alpha)
        Xr = sp.csr_matrix(Xb, dtype=np.float64) @ sp.diags(r)
        svm = LinearSVC(C=self.C, loss="squared_hinge", penalty="l2", dual=True, tol=1e-4, max_iter=self.max_iter,
                        random_state=self.seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            svm.fit(Xr, np.asarray(ybin, np.int64))
        w = svm.coef_.ravel().astype(np.float64)
        w2 = (1.0 - self.beta) * float(np.abs(w).mean()) + self.beta * w
        return r, w2, float(svm.intercept_[0])

    def fit(self, Xb, y) -> "NBSVM":
        y = np.asarray(y, np.int64)
        self.classes_ = np.unique(y)
        if self.classes_.shape[0] < 2:
            self.models_ = []
            return self
        targets = [self.classes_[1]] if self.classes_.shape[0] == 2 else list(self.classes_)
        self.models_ = [self._one(Xb, y == c) for c in targets]
        return self

    def decision_function(self, Xb) -> np.ndarray:
        import scipy.sparse as sp

        Xb = sp.csr_matrix(Xb, dtype=np.float64)
        return np.stack([np.asarray(Xb @ (r * w)).ravel() + b for r, w, b in self.models_], axis=1)

    def predict(self, Xb) -> np.ndarray:
        if not self.models_:
            return np.full(Xb.shape[0], self.classes_[0])
        d = self.decision_function(Xb)
        if self.classes_.shape[0] == 2:
            return self.classes_[(d[:, 0] > 0).astype(np.int64)]
        return self.classes_[np.argmax(d, axis=1)]


def _cv_choice(make: Callable[[Any], Any], grid: Sequence, Xl, yl, seed: int, folds: int) -> tuple[Any, dict, int]:
    splits = TR.cv_splits(np.asarray(yl), seed, folds)
    cv = {}
    for g in grid:
        accs = []
        for tr, va in splits:
            m = make(g).fit(Xl[tr], yl[tr])
            accs.append(float(np.mean(m.predict(Xl[va]) == yl[va])))
        cv[float(g)] = float(np.mean(accs))
    best = max(cv.values())
    return min(g for g, a in cv.items() if a == best), cv, len(splits)


def fit_mnb(Xcount, yl, seed: int, alpha_grid: Sequence[float], folds: int) -> Frozen:
    """MNB with alpha by stratified CV inside the labelled sample (ties to the smaller alpha), refit on the sample."""
    yl = np.asarray(yl, np.int64)
    a, cv, nf = _cv_choice(MNB, alpha_grid, Xcount, yl, seed, folds)
    return Frozen("mnb", MNB(a).fit(Xcount, yl), a, {"folds": nf, "acc": cv})


def fit_nbsvm(Xbin, yl, seed: int, c_grid: Sequence[float], folds: int, alpha: float, beta: float, max_iter: int) -> Frozen:
    """NBSVM with C by stratified CV inside the labelled sample (ties to the smaller C), refit on the sample."""
    yl = np.asarray(yl, np.int64)
    make = (lambda C: NBSVM(C, alpha, beta, seed, max_iter))
    C, cv, nf = _cv_choice(make, c_grid, Xbin, yl, seed, folds)
    return Frozen("nbsvm", make(C).fit(Xbin, yl), C, {"folds": nf, "acc": cv}, extra={"alpha": alpha, "beta": beta})


def frozen_predict(f: Frozen, X) -> np.ndarray:
    if f.kind == "lr":
        return np.asarray(TR._predict_lr(f.model, X, f.dense)).astype(np.int64)
    if f.kind == "tm":
        return tm_predict(f.model, X)
    return np.asarray(f.model.predict(X)).astype(np.int64)


# ====================================================================== 3. statistics (A8 item 1)
T_975 = TR.T_975


def _as_arrays(units: dict) -> tuple[list, list]:
    keys = sorted(units)
    return keys, [np.asarray(units[k], np.float64).reshape(-1) for k in keys]


def two_level_bootstrap(units: dict, n_boot: int = 10000, seed: int = 20260925, ci: float = 0.95, chunk: int = 250) -> dict:
    """Two-level bootstrap of A8 item 1. `units[seed]` holds the unit-level values of one pretraining seed (for a
    paired difference: the per-label-draw differences). Statistic: the mean over seeds of the mean over units. One
    resample: draw S seeds with replacement; for every drawn position draw that seed's units with replacement
    (independently per position). Interval: the percentile interval of the resampled statistics (numpy linear
    quantiles, as tmlm.ssl.probes.paired_bootstrap). Random stream: SeedSequence([seed, 1718, S, n_1..n_S]);
    per chunk: the seed draws (m, S), then for position k = 0..S-1 and seed s = 0..S-1 the unit draws (m, n_s)."""
    keys, arrs = _as_arrays(units)
    S = len(arrs)
    if S == 0 or any(a.shape[0] == 0 for a in arrs):
        return {"mean": None, "lo": None, "hi": None, "excludes_0": None, "n_seeds": S, "n_boot": int(n_boot)}
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 1718, S] + [int(a.shape[0]) for a in arrs]))
    stats = np.empty(int(n_boot))
    for c0 in range(0, int(n_boot), int(chunk)):
        m = min(int(chunk), int(n_boot) - c0)
        pick = rng.integers(0, S, size=(m, S))
        means = np.empty((m, S))
        for k in range(S):
            per = np.empty((m, S))
            for s, a in enumerate(arrs):
                idx = rng.integers(0, a.shape[0], size=(m, a.shape[0]))
                per[:, s] = a[idx].mean(axis=1)
            means[:, k] = per[np.arange(m), pick[:, k]]
        stats[c0:c0 + m] = means.mean(axis=1)
    return _boot_summary(float(np.mean([a.mean() for a in arrs])), stats, ci, keys, arrs)


def two_level_bootstrap_unpaired(units: dict, n_boot: int = 10000, seed: int = 20260925, ci: float = 0.95, chunk: int = 50) -> dict:
    """As two_level_bootstrap for an UNPAIRED difference inside each seed: units[seed] = (A, B) (for Q_int: the
    NPMI of the winner's concept groups and of C0y's); the per-seed statistic is mean(A*) - mean(B*) with A* and B*
    resampled independently. Stream: SeedSequence([seed, 1719, S, nA_1, nB_1, ...]); per chunk and position k,
    seed s: the draws of A (m, nA_s), then of B (m, nB_s)."""
    keys = sorted(units)
    pairs = [(np.asarray(units[k][0], np.float64).reshape(-1), np.asarray(units[k][1], np.float64).reshape(-1)) for k in keys]
    S = len(pairs)
    if S == 0 or any(a.shape[0] == 0 or b.shape[0] == 0 for a, b in pairs):
        return {"mean": None, "lo": None, "hi": None, "excludes_0": None, "n_seeds": S, "n_boot": int(n_boot)}
    sizes = [int(x.shape[0]) for ab in pairs for x in ab]
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 1719, S] + sizes))
    stats = np.empty(int(n_boot))
    for c0 in range(0, int(n_boot), int(chunk)):
        m = min(int(chunk), int(n_boot) - c0)
        pick = rng.integers(0, S, size=(m, S))
        means = np.empty((m, S))
        for k in range(S):
            per = np.empty((m, S))
            for s, (a, b) in enumerate(pairs):
                ia = rng.integers(0, a.shape[0], size=(m, a.shape[0]))
                ib = rng.integers(0, b.shape[0], size=(m, b.shape[0]))
                per[:, s] = a[ia].mean(axis=1) - b[ib].mean(axis=1)
            means[:, k] = per[np.arange(m), pick[:, k]]
        stats[c0:c0 + m] = means.mean(axis=1)
    point = float(np.mean([a.mean() - b.mean() for a, b in pairs]))
    out = _boot_summary(point, stats, ci, keys, [a - 0.0 for a, _ in pairs])
    out["per_seed"] = {str(k): float(a.mean() - b.mean()) for k, (a, b) in zip(keys, pairs)}
    out["n_units"] = {str(k): [int(a.shape[0]), int(b.shape[0])] for k, (a, b) in zip(keys, pairs)}
    return out


def _boot_summary(point: float, stats: np.ndarray, ci: float, keys: list, arrs: list) -> dict:
    a = (1.0 - float(ci)) / 2.0
    lo, hi = float(np.quantile(stats, a)), float(np.quantile(stats, 1.0 - a))
    return {"mean": point, "lo": lo, "hi": hi, "excludes_0": bool(lo > 0 or hi < 0), "n_seeds": len(keys),
            "seeds": [str(k) for k in keys], "n_units": {str(k): int(x.shape[0]) for k, x in zip(keys, arrs)},
            "per_seed": {str(k): float(x.mean()) for k, x in zip(keys, arrs)}, "n_boot": int(stats.shape[0]),
            "boot_sd": float(stats.std(ddof=1)) if stats.shape[0] > 1 else None, "ci": float(ci)}


def seed_t_interval(units: dict) -> dict:
    """Student-t 95 percent interval over the per-seed means (df = seeds - 1)."""
    keys, arrs = _as_arrays(units)
    return TR.t_interval([float(a.mean()) for a in arrs if a.shape[0]])


# ====================================================================== 4. verdicts (registered rules)
MET, NOT_MET, INCOMPLETE = "MET", "NOT MET", "INCOMPLETE"


def _complete(stat: dict | None, seeds_expected: int) -> bool:
    return bool(stat) and stat.get("mean") is not None and int(stat.get("n_seeds", 0)) == int(seeds_expected)


def verdict_min_ci(name: str, boot: dict | None, t: dict | None, threshold: float, seeds_expected: int = 3,
                   rule: str = "") -> dict:
    """'at least <threshold> points, CI excluding 0' (H17.1, H17.2, H17.4, the A8 label components): MET iff the
    point estimate >= threshold AND the two-level bootstrap interval lies above 0 (lo > 0). The t interval is
    reported next to it."""
    if not _complete(boot, seeds_expected):
        return {"name": name, "verdict": INCOMPLETE, "threshold": threshold, "boot": boot, "t": t, "rule": rule}
    ok_mean = boot["mean"] >= threshold
    ok_ci = boot["lo"] > 0
    return {"name": name, "verdict": MET if (ok_mean and ok_ci) else NOT_MET, "threshold": threshold,
            "mean_meets_threshold": bool(ok_mean), "ci_excludes_0": bool(ok_ci),
            "t_interval_excludes_0": bool(t and t.get("lo") is not None and (t["lo"] > 0 or t["hi"] < 0)),
            "boot": boot, "t": t, "rule": rule or f"point >= {threshold} and two-level bootstrap lo > 0"}


def verdict_above_ci(name: str, boot: dict | None, t: dict | None, seeds_expected: int = 3, rule: str = "") -> dict:
    """'above, CI excluding 0' (H17.12): MET iff the point estimate > 0 AND lo > 0."""
    if not _complete(boot, seeds_expected):
        return {"name": name, "verdict": INCOMPLETE, "boot": boot, "t": t, "rule": rule}
    ok = boot["mean"] > 0 and boot["lo"] > 0
    return {"name": name, "verdict": MET if ok else NOT_MET, "boot": boot, "t": t,
            "rule": rule or "point > 0 and two-level bootstrap lo > 0"}


def verdict_min_mean(name: str, per_seed: dict | None, threshold: float, seeds_expected: int = 3, rule: str = "") -> dict:
    """'at least <threshold>' without an interval condition (H17.11): MET iff the mean over seeds >= threshold."""
    vals = [v for v in (per_seed or {}).values() if v is not None]
    if len(vals) != int(seeds_expected):
        return {"name": name, "verdict": INCOMPLETE, "threshold": threshold, "per_seed": per_seed, "rule": rule}
    m = float(np.mean(vals))
    return {"name": name, "verdict": MET if m >= threshold else NOT_MET, "threshold": threshold, "mean": m,
            "per_seed": per_seed, "t": TR.t_interval(vals), "rule": rule or f"mean over seeds >= {threshold}"}


ESTIMATE = "ESTIMATE (no registered threshold)"


def estimate_row(name: str, boot: dict | None, t: dict | None, seeds_expected: int = 3, rule: str = "") -> dict:
    """A reported difference with its intervals and NO verdict (docs/13 A9 item 5: the A8 rows 'minus the
    exposure-matched young bank' and 'minus the best young bank of any arm' have no registered threshold)."""
    return {"name": name, "verdict": ESTIMATE, "complete": _complete(boot, seeds_expected), "boot": boot, "t": t,
            "rule": rule or "estimate with the two-level bootstrap and the seed-level t interval; no MET or NOT MET"}


def verdict_h17_3(qcon_winner: dict, qcon_ppmi: dict, qcon_c0y: dict, seeds_expected: int = 3,
                  min_vs_ppmi: float = 0.0, min_vs_c0y: float = 0.05) -> dict:
    """H17.3: Q_con(winner) - Q_con(PPMI-SVD) >= 0.0 AND Q_con(winner) - Q_con(C0y) >= 0.05, on the means over the
    confirmatory seeds (per-seed values and t intervals reported)."""
    seeds = sorted(set(qcon_winner) & set(qcon_ppmi) & set(qcon_c0y))
    vals = [(qcon_winner[s], qcon_ppmi[s], qcon_c0y[s]) for s in seeds]
    if len(seeds) != int(seeds_expected) or any(v is None for x in vals for v in x):
        return {"name": "H17.3", "verdict": INCOMPLETE, "inputs": {"winner": qcon_winner, "ppmi": qcon_ppmi, "c0y": qcon_c0y}}
    d_ppmi = [w - p for w, p, _ in vals]
    d_c0y = [w - c for w, _, c in vals]
    a, b = float(np.mean(d_ppmi)), float(np.mean(d_c0y))
    return {"name": "H17.3", "verdict": MET if (a >= min_vs_ppmi and b >= min_vs_c0y) else NOT_MET,
            "winner_minus_ppmi": {"mean": a, "per_seed": dict(zip(map(str, seeds), d_ppmi)), "t": TR.t_interval(d_ppmi),
                                  "threshold": min_vs_ppmi, "meets": bool(a >= min_vs_ppmi)},
            "winner_minus_c0y": {"mean": b, "per_seed": dict(zip(map(str, seeds), d_c0y)), "t": TR.t_interval(d_c0y),
                                 "threshold": min_vs_c0y, "meets": bool(b >= min_vs_c0y)},
            "inputs": {"winner": qcon_winner, "ppmi": qcon_ppmi, "c0y": qcon_c0y},
            "rule": f"mean(winner - PPMI-SVD) >= {min_vs_ppmi} and mean(winner - C0y) >= {min_vs_c0y} (Spearman units)"}


def literal_learning_label(components: dict[str, dict], threshold: float = 1.0) -> dict:
    """A8 item 1: 'literal-learning credit' iff the winner beats max(young ladder) AND the twin by >= 1.0 Q_text point
    at 32 labels with BOTH the all-clauses and the dimension-matched code, every interval excluding 0.
    `components` = {"ladder.all", "ladder.dim", "twin.all", "twin.dim"} -> verdict_min_ci results."""
    need = ("ladder.all", "ladder.dim", "twin.all", "twin.dim")
    if any(k not in components or components[k]["verdict"] == INCOMPLETE for k in need):
        return {"label": INCOMPLETE, "components": components}
    ok = all(components[k]["verdict"] == MET for k in need)
    return {"label": "literal-learning credit" if ok else "no literal-learning credit", "credited": bool(ok),
            "components": {k: components[k]["verdict"] for k in need}, "threshold": threshold}


def winner_by_rule(rule: str, seed100_winner: str, top_two: Sequence[str], dev_q: dict[str, float | None],
                   dev_qcon: dict[str, float | None], tie_points: float = 0.5) -> dict:
    """The R1 winner. 'seed100': the arm fixed by the exploratory seed-100 tournament (docs/13 A9 item 1: A3; the
    only rule the confirmatory config accepts). 'dev_seeds_1_3': the higher mean dev-carve Q_text (points, over seeds
    1 to 3) of the top two; a gap below `tie_points` goes to the higher dev Q_con. Both are computed and returned (the
    second is printed, never used); `rule` says which one is THE winner."""
    a, b = list(top_two)
    alt = None
    detail: dict[str, Any] = {"dev_q_text_points": dev_q, "dev_q_con": dev_qcon, "tie_points": tie_points}
    if dev_q.get(a) is not None and dev_q.get(b) is not None:
        gap = float(dev_q[a] - dev_q[b])
        detail["gap_points"] = gap
        if abs(gap) >= tie_points:
            alt = a if gap > 0 else b
        elif dev_qcon.get(a) is not None and dev_qcon.get(b) is not None:
            alt = a if dev_qcon[a] >= dev_qcon[b] else b
            detail["decided_by"] = "Q_con (tie on Q_text)"
    out = {"rule": rule, "seed100": seed100_winner, "dev_seeds_1_3": alt, **detail}
    if rule == "seed100":
        out["winner"] = seed100_winner
    elif rule == "dev_seeds_1_3":
        out["winner"] = alt
    else:
        raise ValueError(rule)
    return out


# ====================================================================== 5. Q_int: NPMI coherence and word intrusion
def cooc_window(tok: np.ndarray, doc_start: np.ndarray, vocab: int, window: int) -> np.ndarray:
    """Dense symmetric (V, V) int64: ordered pairs of positions (p, q) inside one document with 1 <= |p - q| <=
    window - 1 (a sliding window of `window` tokens), X[tok[p], tok[q]] += 1 (tmlm.ssl.concepts.cooc_counts)."""
    from . import concepts as KC

    offs = [o for o in range(-(int(window) - 1), int(window)) if o != 0]
    return KC.cooc_counts(np.asarray(tok), np.asarray(doc_start), int(vocab), offs)


class NPMI:
    """NPMI over a co-occurrence table X (symmetric counts): P(a, b) = X[a, b] / M, P(a) = sum_b X[a, b] / M,
    M = sum X; NPMI = ln(P(a, b) / (P(a) P(b))) / -ln P(a, b); -1 when X[a, b] = 0 (the limit); a == b is 1."""

    def __init__(self, X: np.ndarray) -> None:
        self.X = np.asarray(X)
        self.M = float(self.X.sum())
        self.row = self.X.sum(axis=1).astype(np.float64)

    def __call__(self, a, b) -> np.ndarray:
        a = np.asarray(a, np.int64)
        b = np.asarray(b, np.int64)
        n = self.X[a, b].astype(np.float64)
        out = np.full(np.broadcast(a, b).shape, -1.0)
        ok = n > 0
        if self.M > 0 and np.any(ok):
            pab = n[ok] / self.M
            pa = self.row[np.broadcast_to(a, out.shape)[ok]] / self.M
            pb = self.row[np.broadcast_to(b, out.shape)[ok]] / self.M
            den = -np.log(pab)
            val = np.where(den > 0, np.log(pab / (pa * pb)) / np.where(den > 0, den, 1.0), 1.0)
            out[ok] = val
        out[np.broadcast_to(a == b, out.shape)] = 1.0
        return out


def group_coherence(groups: np.ndarray, npmi: NPMI) -> np.ndarray:
    """float64 [G]: mean NPMI over the unordered word pairs of each group (rows of `groups`, -1 padding ignored;
    a group with fewer than 2 words is NaN)."""
    groups = np.asarray(groups, np.int64)
    out = np.full(groups.shape[0], np.nan)
    n = groups.shape[1]
    ia, ib = np.triu_indices(n, k=1)
    A, B = groups[:, ia], groups[:, ib]
    ok = (A >= 0) & (B >= 0)
    vals = np.where(ok, npmi(np.maximum(A, 0), np.maximum(B, 0)), 0.0)
    cnt = ok.sum(axis=1)
    has = cnt > 0
    out[has] = vals[has].sum(axis=1) / cnt[has]
    return out


def top_words_ppmi(counts, clauses: np.ndarray, word_ok: np.ndarray, n: int, min_count: int, cds: float = 0.75) -> tuple[np.ndarray, np.ndarray]:
    """Concept group of clause j = its n top words: tokens c with word_ok[c] and O_jc >= min_count, ranked by
    PPMI(j, c) = max(0, ln(O_jc Z / (R_j N_c^cds))) descending (ties toward the lower token id), R_j = sum_c O_jc,
    N_c = sum_j O_jc (all clauses of the table), Z = sum_c N_c^cds. `counts`: dense (C, V) or scipy CSR.
    Returns (groups [len(clauses), n] with -1 padding, qualifies [len(clauses)]: at least n candidates)."""
    import scipy.sparse as sp

    Xs = sp.csr_matrix(counts)
    Nc = np.asarray(Xs.sum(axis=0)).ravel().astype(np.float64)
    Rj = np.asarray(Xs.sum(axis=1)).ravel().astype(np.float64)
    Z = float(np.power(Nc[Nc > 0], cds).sum())
    word_ok = np.asarray(word_ok, bool)
    clauses = np.asarray(clauses, np.int64)
    groups = np.full((clauses.shape[0], int(n)), -1, np.int64)
    qual = np.zeros(clauses.shape[0], bool)
    for i, j in enumerate(clauses.tolist()):
        lo, hi = Xs.indptr[j], Xs.indptr[j + 1]
        c = Xs.indices[lo:hi].astype(np.int64)
        v = Xs.data[lo:hi].astype(np.float64)
        keep = word_ok[c] & (v >= min_count)
        c, v = c[keep], v[keep]
        if c.size == 0:
            continue
        w = np.maximum(np.log(v) + math.log(Z) - math.log(Rj[j]) - cds * np.log(Nc[c]), 0.0)
        order = np.lexsort((c, -w))[: int(n)]
        groups[i, : order.shape[0]] = c[order]
        qual[i] = c.shape[0] >= int(n)
    return groups, qual


def intruder_task(groups: np.ndarray, npmi: NPMI, seed: int, n_keep: int = 5) -> dict:
    """Automated word intrusion (Lau, Newman and Baldwin 2014, the NPMI feature): for every group g the first n_keep
    words plus one intruder, a word of the top words of another group h (uniform, h != g) that is not among g's top
    words; detected iff the intruder has the strictly lowest mean NPMI to the other words of the set (a tie counts
    as not detected). Stream SeedSequence([seed, 0x1A7])."""
    groups = np.asarray(groups, np.int64)
    G = groups.shape[0]
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x1A7]))
    det, tried = [], 0
    for g in range(G):
        base = [int(x) for x in groups[g, :n_keep] if x >= 0]
        if len(base) < 2 or G < 2:
            det.append(np.nan)
            continue
        h = int(rng.integers(0, G - 1))
        h = h + 1 if h >= g else h
        own = set(int(x) for x in groups[g] if x >= 0)
        cand = [int(x) for x in groups[h] if x >= 0 and int(x) not in own]
        if not cand:
            det.append(np.nan)
            continue
        intr = cand[int(rng.integers(0, len(cand)))]
        words = np.asarray(base + [intr], np.int64)
        M = npmi(words[:, None], words[None, :])
        np.fill_diagonal(M, 0.0)
        score = M.sum(axis=1) / (words.shape[0] - 1)
        lo = score.min()
        tried += 1
        det.append(float(score[-1] == lo and int((score == lo).sum()) == 1))
    d = np.asarray(det, np.float64)
    ok = ~np.isnan(d)
    return {"accuracy": float(d[ok].mean()) if ok.any() else None, "n_groups_scored": int(ok.sum()),
            "chance": 1.0 / (n_keep + 1), "detected": d}


# ====================================================================== 6. random conjunction bank (A8 item 1)
@dataclass
class RandomBank:
    """col[j, t] = G+ column (offset, kind) of literal t of clause j (-1 padding), sym[j, t] its symbol; n_lit[j]."""
    mode: str
    vocab: int
    col: np.ndarray
    sym: np.ndarray
    n_lit: np.ndarray
    seed: int
    info: dict

    @property
    def n(self) -> int:
        return int(self.col.shape[0])

    def save(self, path) -> None:
        np.savez(path, col=self.col, sym=self.sym, n_lit=self.n_lit, mode=self.mode, vocab=np.int64(self.vocab),
                 seed=np.int64(self.seed), info=json.dumps(self.info))

    @staticmethod
    def load(path) -> "RandomBank":
        z = np.load(path)
        return RandomBank(str(z["mode"]), int(z["vocab"]), z["col"], z["sym"], z["n_lit"], int(z["seed"]), json.loads(str(z["info"])))

    def keys(self, ns: int) -> np.ndarray:
        """int64 [C, 3]: literal keys col * ns + sym (-1 padding)."""
        return np.where(self.col >= 0, self.col * int(ns) + self.sym, -1)


def _col_tables(mode: str) -> tuple[np.ndarray, np.ndarray]:
    from . import controls as K

    cols = K.columns(mode)
    return np.array([o for o, _ in cols], np.int64), np.array([k for _, k in cols], np.int64)


def symbol_at(tok: np.ndarray, lo: np.ndarray, hi: np.ndarray, kind_map: np.ndarray, p: np.ndarray, o: np.ndarray,
              k: np.ndarray, vocab: int) -> np.ndarray:
    """int64: the symbol of column (o, k) at positions p (per element), -1 where the column has no true literal
    (the rule of tmlm.ssl.controls.sym_column: outside the document only kind 0 carries SYM_EDGE)."""
    from ..engine.config import sym_edge

    q = p + o
    inside = (q >= lo[p]) & (q < hi[p])
    t = np.asarray(tok)[np.clip(q, 0, np.asarray(tok).shape[0] - 1)].astype(np.int64)
    s = np.asarray(kind_map, np.int64)[k, t]
    return np.where(inside, s, np.where(k == 0, int(sym_edge(vocab)), -1))


def random_bank(tok: np.ndarray, doc_start: np.ndarray, kind_map: np.ndarray, mode: str, vocab: int, n_clauses: int,
                seed: int, max_lit: int = 3) -> RandomBank:
    """A8 random conjunction bank (the reading fixed in the config): for every clause n_lit uniform on {1, .., max_lit};
    literal t of the clause: draw a position p uniformly from the pretraining prefix and a column (offset, kind)
    uniformly among the columns whose offset the clause does not use yet (G+ literal space: offsets of the mode,
    kinds 0..3); the literal is the symbol true at (p, column); a draw without a true literal (kind > 0 outside the
    document) is redrawn. Every true (position, column) pair is equally likely, so a literal is drawn with
    probability proportional to its frequency in the prefix; literals of one clause are drawn independently.
    Stream SeedSequence([20260925, 0xA8, 3, seed])."""
    from . import controls as K

    tok = np.asarray(tok)
    N = int(tok.shape[0])
    lo, hi = K.doc_bounds(np, np.asarray(doc_start))
    off, kind = _col_tables(mode)
    n_cols = off.shape[0]
    rng = np.random.default_rng(np.random.SeedSequence([20260925, 0xA8, 3, int(seed)]))
    C = int(n_clauses)
    n_lit = rng.integers(1, int(max_lit) + 1, size=C).astype(np.int64)
    col = np.full((C, int(max_lit)), -1, np.int64)
    sym = np.full((C, int(max_lit)), -1, np.int64)
    redraws = 0
    for t in range(int(max_lit)):
        pending = np.flatnonzero(n_lit > t)
        while pending.size:
            p = rng.integers(0, N, size=pending.size)
            c = rng.integers(0, n_cols, size=pending.size)
            used = np.where(col[pending, :t] >= 0, off[np.maximum(col[pending, :t], 0)], 1 << 30) if t else np.zeros((pending.size, 0), np.int64)
            clash = np.any(used == off[c][:, None], axis=1) if t else np.zeros(pending.size, bool)
            s = symbol_at(tok, lo, hi, kind_map, p, off[c], kind[c], vocab)
            ok = ~clash & (s >= 0)
            col[pending[ok], t] = c[ok]
            sym[pending[ok], t] = s[ok]
            redraws += int((~ok).sum())
            pending = pending[~ok]
    info = {"n_clauses": C, "max_lit": int(max_lit), "n_lit_counts": np.bincount(n_lit, minlength=int(max_lit) + 1)[1:].tolist(),
            "redraws": redraws, "rule": "position uniform, column uniform among unused offsets, redraw when no literal is true"}
    return RandomBank(mode, int(vocab), col, sym, n_lit, int(seed), info)


def literal_single_counts(xp, tok, doc_start, kind_map, mode: str, vocab: int, chunk: int = 1 << 22) -> np.ndarray:
    """int64 [n_cols, n_sym]: number of positions where literal (column, symbol) is true (the G+ singles)."""
    from ..engine.config import n_sym
    from . import controls as K

    ns = int(n_sym(vocab))
    tok_d = xp.asarray(np.asarray(tok))
    km_d = xp.asarray(np.asarray(kind_map)).astype(xp.int64)
    lo, hi = K.doc_bounds(xp, np.asarray(doc_start))
    cols = K.columns(mode)
    out = np.zeros((len(cols), ns), np.int64)
    n = int(tok_d.shape[0])
    for a in range(0, n, int(chunk)):
        p = xp.arange(a, min(n, a + int(chunk)), dtype=xp.int64)
        for ci, (o, k) in enumerate(cols):
            v = K.sym_column(xp, tok_d, lo, hi, km_d, p, o, k, vocab)
            v = v[v >= 0]
            if v.size:
                b = xp.bincount(v.astype(xp.int64), minlength=ns)
                out[ci] += b.get() if hasattr(b, "get") else np.asarray(b)
    return out


def anchor_order(bank: RandomBank, single: np.ndarray) -> np.ndarray:
    """int64 [C, 3]: literal slots of every clause reordered so that slot 0 is the literal with the smallest
    single count (the anchor; ties toward the lower slot), padding last."""
    cnt = np.where(bank.col >= 0, single[np.maximum(bank.col, 0), np.maximum(bank.sym, 0)], np.iinfo(np.int64).max)
    return np.argsort(cnt, axis=1, kind="stable")


def _expand(xp, starts, lens):
    """Flattened ranges [starts_i, starts_i + lens_i) with the owner index i (device-side: cumsum and searchsorted,
    no host round trip of the lengths)."""
    if lens.shape[0] == 0:
        return xp.zeros(0, xp.int64), xp.zeros(0, xp.int64)
    ends = xp.cumsum(lens)
    tot = int(ends[-1])
    if tot == 0:
        return xp.zeros(0, xp.int64), xp.zeros(0, xp.int64)
    idx = xp.arange(tot, dtype=xp.int64)
    owner = xp.searchsorted(ends, idx, side="right").astype(xp.int64)
    return starts[owner] + (idx - (ends - lens)[owner]), owner


class RandomBankFirer:
    """Fires a RandomBank over a token corpus: per chunk of positions the G+ symbol matrix, candidates from the
    anchor-literal index, then the check of the remaining literals. xp = numpy or cupy (identical integers)."""

    def __init__(self, xp, bank: RandomBank, single: np.ndarray, kind_map: np.ndarray) -> None:
        from ..engine.config import n_sym

        self.xp, self.bank = xp, bank
        self.ns = int(n_sym(bank.vocab))
        o = anchor_order(bank, single)
        rows = np.arange(bank.n)[:, None]
        keys = bank.keys(self.ns)[rows, o]
        self.col = bank.col[rows, o]
        self.sym = bank.sym[rows, o]
        order = np.argsort(keys[:, 0], kind="stable")
        ak = keys[order, 0]
        uk, first, cnt = np.unique(ak, return_index=True, return_counts=True)
        self.akeys, self.astart, self.alen = xp.asarray(uk), xp.asarray(first.astype(np.int64)), xp.asarray(cnt.astype(np.int64))
        self.aclause = xp.asarray(order.astype(np.int64))
        self.col_d, self.sym_d = xp.asarray(self.col), xp.asarray(self.sym)
        self.km = xp.asarray(np.asarray(kind_map)).astype(xp.int64)

    def fire(self, tok_d, lo, hi, pos) -> tuple[Any, Any]:
        """(row index into pos, clause id) of every firing, for the positions `pos` (xp int64)."""
        from . import controls as K

        xp = self.xp
        S = K.sym_matrix(xp, tok_d, lo, hi, self.km, pos, self.bank.mode, self.bank.vocab).astype(xp.int64)
        m, nc = S.shape
        keys = xp.arange(nc, dtype=xp.int64)[None, :] * self.ns + S
        valid = S >= 0
        r, c = xp.nonzero(valid)
        k = keys[r, c]
        if self.akeys.shape[0] == 0 or k.size == 0:
            return xp.zeros(0, xp.int64), xp.zeros(0, xp.int64)
        i = xp.minimum(xp.searchsorted(self.akeys, k), self.akeys.shape[0] - 1)
        hit = self.akeys[i] == k
        r, i = r[hit], i[hit]
        flat, owner = _expand(xp, self.astart[i], self.alen[i])
        rows = r[owner]
        cl = self.aclause[flat]
        ok = xp.ones(cl.shape[0], dtype=bool)
        for t in range(1, self.col.shape[1]):
            ct = self.col_d[cl, t]
            has = ct >= 0
            ok &= ~has | (S[rows, xp.maximum(ct, 0)] == self.sym_d[cl, t])
        return rows[ok], cl[ok]


def random_bank_doc_counts(xp, firer: RandomBankFirer, tok, doc_start, chunk: int = 4096):
    """SparseDocs: per document the number of its positions where each clause fires (the count code of the banks)."""
    from . import controls as K
    from . import features as F

    ds = np.asarray(doc_start, np.int64)
    tok_d = xp.asarray(np.asarray(tok))
    lo, hi = K.doc_bounds(xp, ds)
    parts = []
    for d0, d1 in K.doc_chunks(ds, int(chunk)):
        a, b = int(ds[d0]), int(ds[d1])
        pos = xp.arange(a, b, dtype=xp.int64)
        rows, cl = firer.fire(tok_d, lo, hi, pos)
        doc_rel = xp.asarray(np.repeat(np.arange(d1 - d0, dtype=np.int64), np.diff(ds[d0:d1 + 1])))
        key = (doc_rel[rows] << 32) | cl
        u, cnt = xp.unique(key, return_counts=True)
        uh = u.get() if hasattr(u, "get") else np.asarray(u)
        ch = cnt.get() if hasattr(cnt, "get") else np.asarray(cnt)
        drel = (uh >> 32).astype(np.int64)
        ip = np.zeros(d1 - d0 + 1, np.int64)
        ip[1:] = np.cumsum(np.bincount(drel, minlength=d1 - d0))
        parts.append(F.SparseDocs(ip, (uh & 0xFFFFFFFF).astype(np.uint32), ch.astype(np.uint32), firer.bank.n))
    if not parts:
        return F.SparseDocs(np.zeros(1, np.int64), np.zeros(0, np.uint32), np.zeros(0, np.uint32), firer.bank.n)
    return F.merge_rows(parts)


def random_bank_fire_counts(xp, firer: RandomBankFirer, tok, doc_start, chunk: int = 4096) -> np.ndarray:
    """u64 [C]: number of corpus positions where each clause fires."""
    from . import controls as K

    tok_d = xp.asarray(np.asarray(tok))
    lo, hi = K.doc_bounds(xp, np.asarray(doc_start))
    n = int(tok_d.shape[0])
    acc = xp.zeros(firer.bank.n, dtype=xp.int64)
    for a in range(0, n, int(chunk)):
        _, cl = firer.fire(tok_d, lo, hi, xp.arange(a, min(n, a + int(chunk)), dtype=xp.int64))
        if cl.size:
            acc += xp.bincount(cl, minlength=firer.bank.n)
    out = acc.get() if hasattr(acc, "get") else np.asarray(acc)
    return out.astype(np.uint64)


def random_bank_bruteforce(bank: RandomBank, tok, doc_start, kind_map, positions) -> list[list[int]]:
    """Reference (pure Python, tiny corpora): the firing clause ids at every position of `positions`."""
    from ..engine.config import sym_edge

    tok = [int(t) for t in np.asarray(tok)]
    ds = [int(x) for x in np.asarray(doc_start)]
    km = np.asarray(kind_map)
    off, kind = _col_tables(bank.mode)
    doc_of = np.repeat(np.arange(len(ds) - 1), np.diff(ds))
    edge = int(sym_edge(bank.vocab))
    out = []
    for p in positions:
        d = int(doc_of[p])
        fired = []
        for j in range(bank.n):
            ok = True
            for t in range(bank.col.shape[1]):
                c = int(bank.col[j, t])
                if c < 0:
                    continue
                q = p + int(off[c])
                if ds[d] <= q < ds[d + 1]:
                    s = int(km[int(kind[c])][tok[q]])
                else:
                    s = edge if int(kind[c]) == 0 else -1
                if s != int(bank.sym[j, t]):
                    ok = False
                    break
            if ok:
                fired.append(j)
        out.append(fired)
    return out


# ====================================================================== 7. Q_con: the A7 word code
def word_runs(word_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(start, length) of every maximal run of equal word_id (NO_WORD tokens excluded), ascending."""
    from ..data.corpus_store import NO_WORD

    w = np.asarray(word_id, np.int64)
    n = w.shape[0]
    if n == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    brk = np.flatnonzero(w[1:] != w[:-1]) + 1
    starts = np.concatenate([[0], brk]).astype(np.int64)
    lens = np.diff(np.concatenate([starts, [n]])).astype(np.int64)
    keep = w[starts] != int(NO_WORD)
    return starts[keep], lens[keep]


def find_occurrences(tok: np.ndarray, starts: np.ndarray, lens: np.ndarray, targets: dict[str, Sequence[int]]) -> dict[str, np.ndarray]:
    """word -> ascending start positions of the runs whose tokens equal the word's token ids exactly."""
    tok = np.asarray(tok, np.int64)
    first = tok[starts]
    key = lens * (1 << 20) + first
    order = np.argsort(key, kind="stable")
    ks = key[order]
    out: dict[str, np.ndarray] = {}
    for w, ids in targets.items():
        ids = [int(x) for x in ids]
        if not ids:
            out[w] = np.zeros(0, np.int64)
            continue
        k = len(ids) * (1 << 20) + ids[0]
        a, b = np.searchsorted(ks, [k, k + 1])
        cand = np.sort(starts[order[a:b]])
        ok = np.ones(cand.shape[0], bool)
        for i in range(1, len(ids)):
            ok &= tok[cand + i] == ids[i]
        out[w] = cand[ok]
    return out


def select_occurrences(starts: np.ndarray, k: int) -> np.ndarray:
    """At most k occurrences: the ones with the smallest mix64(start position) (a fixed hash sample spread over the
    prefix; mix64 is a bijection, so there are no ties), returned in ascending position order."""
    s = np.asarray(starts, np.int64)
    if s.shape[0] <= int(k):
        return np.sort(s)
    h = mix64_np(s.astype(np.uint64))
    return np.sort(s[np.argsort(h, kind="stable")[: int(k)]])


def occurrence_rounds(starts: np.ndarray, doc_of_pos: np.ndarray) -> np.ndarray:
    """int64: the round of every occurrence = its rank among the selected occurrences of its document (by start).
    In one round every document holds at most one hidden occurrence, so no occurrence ever sees another one hidden
    (masked and diffusion contexts, including bag literals, never cross a document)."""
    s = np.asarray(starts, np.int64)
    d = np.asarray(doc_of_pos)[s].astype(np.int64)
    order = np.lexsort((s, d))
    ds = d[order]
    first = np.ones(ds.shape[0], bool)
    first[1:] = ds[1:] != ds[:-1]
    grp = np.maximum.accumulate(np.where(first, np.arange(ds.shape[0]), 0))
    r = np.empty(s.shape[0], np.int64)
    r[order] = np.arange(ds.shape[0]) - grp
    return r


@dataclass
class Occurrences:
    """Selected occurrences of the words: word index, start, length (all int64, grouped by word)."""
    words: list[str]
    widx: np.ndarray
    start: np.ndarray
    length: np.ndarray
    n_total: np.ndarray       # all occurrences of each word in the prefix (coverage)


def build_occurrences(tok, word_id, targets: dict[str, Sequence[int]], max_occ: int) -> Occurrences:
    starts, lens = word_runs(word_id)
    found = find_occurrences(tok, starts, lens, targets)
    words = list(targets)
    w_i, st, ln, tot = [], [], [], []
    for i, w in enumerate(words):
        allw = found[w]
        tot.append(int(allw.shape[0]))
        sel = select_occurrences(allw, max_occ)
        w_i.append(np.full(sel.shape[0], i, np.int64))
        st.append(sel)
        ln.append(np.full(sel.shape[0], len(targets[w]), np.int64))
    cat = (lambda x: np.concatenate(x) if x else np.zeros(0, np.int64))
    return Occurrences(words, cat(w_i), cat(st), cat(ln), np.asarray(tot, np.int64))


def word_codes_api(eng, occ: Occurrences, n_tok: int, rounds: np.ndarray, epoch: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Word code counts through the engine's public fired_clauses (any engine; tests and small runs): per round one
    hidden bitmap with the subtokens of that round's occurrences; per occurrence the union of the clauses fired at
    its subtoken positions. Returns COO (word index, clause id, count of occurrences)."""
    ev_w, ev_j = [], []
    for r in range(int(rounds.max()) + 1 if rounds.size else 0):
        sel = np.flatnonzero(rounds == r)
        hidden = np.zeros(int(n_tok), np.uint8)
        for i in sel.tolist():
            hidden[occ.start[i]:occ.start[i] + occ.length[i]] = 1
        for i in sel.tolist():
            u = set()
            for q in range(int(occ.start[i]), int(occ.start[i] + occ.length[i])):
                u.update(int(x) for x in eng.fired_clauses(epoch, q, hidden))
            if u:
                ev_w.append(np.full(len(u), occ.widx[i], np.int64))
                ev_j.append(np.fromiter(sorted(u), np.int64))
    return _coo_counts(ev_w, ev_j)


def word_codes_cuda(eng, occ: Occurrences, n_tok: int, rounds: np.ndarray, epoch: int = 0, max_positions: int = 1 << 18) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """word_codes_api on the CUDA engine: one hidden device bitmap per round (set, forward, reset), the forward of
    tmlm.engine.cuda (the R1 featurisation path), unique (occurrence, clause) pairs, then counts per (word, clause)."""
    cp = eng.cp
    hid = cp.zeros(int(n_tok), dtype=cp.uint8)
    ev_w, ev_j = [], []
    n_r = int(rounds.max()) + 1 if rounds.size else 0
    for r in range(n_r):
        sel = np.flatnonzero(rounds == r)
        if sel.size == 0:
            continue
        L = occ.length[sel]
        pos = np.repeat(occ.start[sel], L) + (np.arange(int(L.sum())) - np.repeat(np.cumsum(L) - L, L))
        occ_of = np.repeat(sel, L)
        pos_d = cp.asarray(pos.astype(np.int64))
        hid[pos_d] = 1
        keys = []
        for a in range(0, pos.shape[0], int(max_positions)):
            b = min(pos.shape[0], a + int(max_positions))
            eng._sync_state_struct()
            fw = eng._forward(epoch, cp.asarray(pos[a:b].astype(np.uint32)), hid, 0)
            eng._marks = []
            fe = fw.fired_ex.get().astype(np.int64)
            fj = fw.fired_j.get().astype(np.int64)
            del fw
            if fe.size:
                keys.append((occ_of[a:b][fe] << 32) | fj)
        hid[pos_d] = 0
        if keys:  # unique per (occurrence, clause) over the whole round: an occurrence may span two position blocks
            u = np.unique(np.concatenate(keys))
            ev_w.append(occ.widx[(u >> 32)])
            ev_j.append(u & 0xFFFFFFFF)
    eng.synchronize()
    return _coo_counts(ev_w, ev_j)


def _coo_counts(ev_w: list, ev_j: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not ev_w:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64)
    w = np.concatenate(ev_w).astype(np.int64)
    j = np.concatenate(ev_j).astype(np.int64)
    u, c = np.unique((w << 32) | j, return_counts=True)
    return (u >> 32).astype(np.int64), (u & 0xFFFFFFFF).astype(np.int64), c.astype(np.int64)


def restrict_columns(w: np.ndarray, j: np.ndarray, c: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """COO counts restricted to the columns `cols` (the dimension-matched clauses) and renumbered 0..len(cols)-1 in
    the order of `cols` (the dimension-matched word code, reported next to the all-clauses code, A9 item 2)."""
    cols = np.asarray(cols, np.int64)
    if np.unique(cols).shape[0] != cols.shape[0]:
        raise ValueError("duplicate columns")
    w, j, c = np.asarray(w, np.int64), np.asarray(j, np.int64), np.asarray(c, np.int64)
    if cols.shape[0] == 0:
        return w[:0], j[:0], c[:0]
    order = np.argsort(cols, kind="stable")
    sc = cols[order]
    k = np.minimum(np.searchsorted(sc, j), sc.shape[0] - 1)
    keep = sc[k] == j
    return w[keep], order[k[keep]].astype(np.int64), c[keep]


def ppmi_rows(w: np.ndarray, j: np.ndarray, c: np.ndarray, n_words: int, n_cols: int):
    """scipy CSR of the PPMI-weighted word x clause counts (tmlm.ssl.concepts.weight_values 'ppmi', context
    distribution smoothing 0.75), rows L2-normalised (cosine by dot products)."""
    import scipy.sparse as sp

    from . import concepts as KC

    if w.size == 0:
        return sp.csr_matrix((n_words, n_cols))
    m = KC.marginals_from_coo(w, j, c, n_words, n_cols, np.full(n_words, 1.0 / max(1, n_words)))
    val = KC.weight_values(np, w, j, c.astype(np.float64), m.to(np), "ppmi").astype(np.float64)
    X = sp.csr_matrix((val, (w, j)), shape=(n_words, n_cols))
    X.eliminate_zeros()
    nrm = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    nrm[nrm == 0] = 1.0
    return (sp.diags(1.0 / nrm) @ X).tocsr()


def pair_cosines_sparse(X, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine of rows a[i] and b[i] of an L2-normalised CSR (0 for an empty row)."""
    return np.asarray(X[np.asarray(a, np.int64)].multiply(X[np.asarray(b, np.int64)]).sum(axis=1)).ravel()


def word_vectors_mean(E: np.ndarray, token_ids: Sequence[Sequence[int]]) -> np.ndarray:
    """Word vector = mean of the token vectors of its subtokens (PPMI-SVD, CBOW), rows L2-normalised."""
    E = np.asarray(E, np.float64)
    W = np.stack([E[np.asarray(ids, np.int64)].mean(axis=0) for ids in token_ids])
    n = np.linalg.norm(W, axis=1)
    n[n == 0] = 1.0
    return W / n[:, None]


def q_con_from_similarity(sets: dict[str, dict], sim: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> dict:
    """Spearman per dataset over its covered pairs (word indices a, b; gold) and the mean over datasets (Q_con)."""
    from . import concepts as KC

    out = {}
    for name, s in sets.items():
        if s["a"].shape[0] < 3:
            out[name] = None
            continue
        out[name] = KC.spearman(np.asarray(s["gold"], np.float64), sim(s["a"], s["b"]))
    vals = [v for v in out.values()]
    out["q_con"] = float(np.mean(vals)) if vals and all(v is not None for v in vals) else None
    return out


__all__ = ["load_confirm", "load_men_test", "record_confirm_access", "prior_accesses", "own_accesses",
           "ConfirmAccessRefused", "ConfirmPreflightFailed", "test_file_metadata", "repo_file_metadata",
           "write_test_hash_reference", "verify_or_write_test_hash", "test_hash_reference", "verify_test_file", "git_blob_sha1",
           "NON_LFS_MODES",
           "sst2_confirm_reference", "preflight_references", "men_test_reference",
           "Frozen", "fit_lr", "fit_tm", "fit_mnb", "fit_nbsvm", "frozen_predict", "MNB", "NBSVM", "nb_log_ratio",
           "two_level_bootstrap", "two_level_bootstrap_unpaired", "seed_t_interval", "verdict_min_ci",
           "verdict_above_ci", "verdict_min_mean", "verdict_h17_3", "estimate_row", "ESTIMATE", "literal_learning_label",
           "winner_by_rule", "cooc_window", "NPMI", "group_coherence", "top_words_ppmi", "intruder_task", "RandomBank",
           "random_bank", "literal_single_counts", "RandomBankFirer", "random_bank_doc_counts", "random_bank_fire_counts",
           "random_bank_bruteforce", "word_runs", "find_occurrences", "select_occurrences", "occurrence_rounds",
           "Occurrences", "build_occurrences", "word_codes_api", "word_codes_cuda", "restrict_columns", "ppmi_rows",
           "pair_cosines_sparse", "word_vectors_mean", "q_con_from_similarity"]
