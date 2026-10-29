"""H8.3 and H8.4 of docs/13 amendment A3: the frozen CVTM-core on the WikiText-103 TEST split, once.

H8.3 (docs/13 line 178): test bits per byte of the CVTM-core, once, after freezing, at most 1.3219
(KN-5 test 1.3719 minus 0.05). H8.4 (line 179): the CVTM-core lower than KN-5 plus a document unigram
cache fitted by the same protocol (components S+D0, docs/11 11.6). docs/13 section 5: every test number
is also reported on the de-overlapped subset (docs/06: a document is kept iff fewer than half of its
13-grams occur in train). A3 lines 188-190: the core contains KN-5, so H8.3 credits counting and
in-context copying in clause form, not Tsetlin learning. The plan of the read is the docs/13 amendment
whose heading carries the words of the config key `test.amendment_heading` (draft:
docs/drafts/H8_3_H8_4_test_plan_DRAFT_2026-10-25.md).

Both splits go through the same code; the validation pass always runs first. With `--split validation`
(the smoke) nothing else happens. With `--split test` the test split is opened only when every check
below passes and every refusal condition is absent; then `protocol.load_eval_split("test", purpose=...)`
appends exactly one row to the shared access log.

Checks before the read (a failure refuses the test read; in a validation run it means exit 1):
  code.cvtm_sources_published        sha256 of src/tmlm/cvtm/*.py = the published pins of the config
  code.cvtm_sources_equal_freeze_without_docstrings
                                     the same files, docstrings removed (AST), = the files of the freeze
  freeze.cache_meta_sha256, freeze.core_file_sha256, freeze.core_state_hash
  config.bootstrap_matches_protocol  the config's bootstrap block = tmlm.cvtm.protocol (2,000, seed)
  data.store_digests                 meta.json digests of the train, validation and test stores = config
  data.one_data_root                 stores, M2 flags and the access log under one data root (and, when
                                     the config says so, that root is the study's data/, TMLM_DATA_DIR unset)
  validation.raw_doc_bytes_sha256    the per-document byte offsets of validation = config
  validation.doc_keep_equals_M2_flags, reproduces.validation_cvtm_core, reproduces.validation_kn5_D0,
  reproduces.validation_kn5_knmodel, validation.kn5_vote_form_within_1e-9_of_knmodel
Refusal conditions of the test read (each is named in the metric `test.refused`):
  a failed check above; --confirm-test-read not given; not run by scripts/run_logged.py under the run
  name of the config (TMLM_RUN_DIR/run.json, status running); no docs/13 heading line ("### A...") with
  the heading words; access-log rows or sha256 other than the plan's; an earlier row with this purpose.
Checks after the read. Invalidating (both verdicts read "invalid", the numbers are kept):
  test.one_access_row_appended, test.access_row_is_this_run (caller, purpose, pid, time),
  test.raw_doc_bytes_sha256, reproduces.test_kn5_knmodel (KNModel path = the CP anchor's number).
Reporting only (the verdicts stay valid; the rows they concern are labelled unverified):
  test.doc_keep_equals_M2_flags (the de-overlapped rows), test.kn5_vote_form_within_1e-9_of_knmodel
  (the vote-form KN-5 row and the pairs that use it).
Exit status: 0 every check passed; 1 at least one check failed (`checks_failed` names them; after a test
read the verdict fields say valid or invalid); 3 the test read was refused, no access row was written;
4 a crash after the read began (`test.crashed_after_read`: the numbers written so far are kept, both
verdicts read "invalid", and the split is not read again). The metrics are written after the access-row
checks, again after the full-split rows and the verdicts, and at the end.

Outputs: metrics (every row on the full split and on the de-overlapped documents: bits per byte, bits per
token, tokens, bytes, document bootstrap intervals, paired differences), the verdicts, the access row of
the read, and `scores_<split>.npz` per split (per-document bits of every row in float64, per-token log2 P,
doc_start, doc_tokens, doc_bytes, doc_keep, doc_frac) with its sha256 in the metrics, so that the
numbers can be recomputed later without another read of the split.

    smoke:   the launch line of the config with --split validation (or --out-dir <scratch> outside the logger)
    test:    the launch line of the config, after the plan is frozen into docs/13
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tmlm.common import hashing  # noqa: E402
from tmlm.cvtm import cache as C  # noqa: E402
from tmlm.cvtm import protocol  # noqa: E402
from tmlm.cvtm.doc_clauses import doc_features  # noqa: E402
from tmlm.cvtm.mixture import D_ORDERS  # noqa: E402
from tmlm.cvtm.model import PTR_MAX, PTR_MIN, suffix_gold  # noqa: E402
from tmlm.data import contamination, corpus_store, evalsets  # noqa: E402
from tmlm.eval.baselines.ngram import bits_per_byte  # noqa: E402

PREREG = ROOT / "docs" / "13-preregistration.md"
ROWS = (("cvtm_core", "full"), ("kn5_D0", "S+D0"), ("kn5", "S"))
KNMODEL = "kn5_knmodel"
ALL_ROWS = tuple(name for name, _ in ROWS) + (KNMODEL,)
PAIRS = (("cvtm_core", "kn5_D0"), ("cvtm_core", "kn5"), ("kn5_D0", "kn5"), ("cvtm_core", KNMODEL))
EXIT_OK, EXIT_CHECK_FAILED, EXIT_REFUSED, EXIT_CRASHED_AFTER_READ = 0, 1, 3, 4
REPORTING_ONLY = ("test.doc_keep_equals_M2_flags", "test.kn5_vote_form_within_1e-9_of_knmodel")


def m(value, label: str = "measured", **extra) -> dict:
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    return {"value": value, "label": label, **extra}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def peak_rss_gib() -> float | None:
    """VmHWM of this process (the peak resident memory), in GiB."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return round(int(line.split()[1]) / 1024.0 ** 2, 2)
    except OSError:
        pass
    return None


# ------------------------------------------------------------------ code and data identity
def code_sha256(path: Path) -> str:
    """sha256 of a module's AST with every docstring removed (comments are not in the AST)."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode("utf-8")).hexdigest()


def code_checks(cfg: dict) -> list[tuple[str, bool, dict]]:
    d = ROOT / "src" / "tmlm" / "cvtm"
    pub = cfg["frozen"]["cvtm_sources_sha256"]
    frz = cfg["frozen"]["cvtm_sources_code_sha256_at_freeze"]
    got = {k: hashing.sha256_file(d / k) for k in pub}
    code = {k: code_sha256(d / k) for k in frz}
    return [("code.cvtm_sources_published", got == pub, {"changed": [k for k in pub if got[k] != pub[k]]}),
            ("code.cvtm_sources_equal_freeze_without_docstrings", code == frz,
             {"changed": [k for k in frz if code[k] != frz[k]]})]


def freeze_file_checks(cfg: dict) -> list[tuple[str, bool, dict]]:
    fz = cfg["frozen"]
    cdir = ROOT / fz["cache_dir"]
    return [("freeze.cache_meta_sha256", hashing.sha256_file(cdir / C.META_FILE) == fz["cache_meta_sha256"], {}),
            ("freeze.core_file_sha256", hashing.sha256_file(cdir / C.CORE_FILE) == fz["core_file_sha256"], {})]


def load_frozen_core(cfg: dict):
    return C.load_core(ROOT / cfg["frozen"]["cache_dir"], backend="numpy")


def data_checks(cfg: dict, splits: tuple[str, ...]) -> tuple[list[tuple[str, bool, dict]], dict]:
    vocab = int(cfg["vocab"])
    got = protocol.dataset_hashes(splits, protocol.CORPUS, vocab)
    want = {k: v for k, v in cfg["frozen"]["data_digests"].items() if k.split("_")[0] in splits}
    root = corpus_store.data_root().resolve()
    paths = {"access_log": evalsets.access_log_path(), "stores": corpus_store.split_dir(protocol.CORPUS, vocab, "train"),
             "raw": corpus_store.raw_dir(protocol.CORPUS, "validation"),
             "m2_flags": contamination.contamination_dir(protocol.CORPUS, vocab)}
    one_root = all(root in Path(p).resolve().parents for p in paths.values())
    project = (not cfg["test"].get("require_project_data_root", True)) or (
        root == (ROOT / "data").resolve() and not os.environ.get("TMLM_DATA_DIR"))
    return ([("data.store_digests", got == want, {"differ": [k for k in want if got.get(k) != want[k]]}),
             ("data.one_data_root", one_root and project, {"data_root": str(root),
                                                           "TMLM_DATA_DIR": os.environ.get("TMLM_DATA_DIR")})],
            got)


def raw_doc_bytes_sha256(split: str) -> str:
    return hashing.sha256_array(np.load(corpus_store.raw_dir(protocol.CORPUS, split) / "doc_bytes.npy"))


def access_rows() -> list[dict]:
    path = evalsets.access_log_path()
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def access_log_sha256() -> str | None:
    path = evalsets.access_log_path()
    return hashing.sha256_file(path) if path.exists() else None


# ------------------------------------------------------------------ scoring
def score_rows(core, variants: dict, ev) -> dict:
    """{row: {"bits_per_byte", "log2p", "doc_bits", ...}} for the three core rows and the KNModel path."""
    lnp, ks = suffix_gold(core.suffix, ev.tok, ev.doc_start)
    feats = doc_features(np.asarray(ev.tok).astype(np.int64), ev.doc_start, orders=D_ORDERS, window=core.window,
                         ptr_min=PTR_MIN, ptr_max=PTR_MAX)
    out = {}
    for name, _ in ROWS:
        c = variants[name]
        rec = c.gold(ev.tok, ev.doc_start, lnp_s=lnp, kstar=ks, feats=feats)
        out[name] = c.score_stream(ev.tok, ev.doc_start, n_bytes=ev.n_bytes, doc_bytes=ev.doc_bytes, rec=rec)
    lp = core.kn.logprob(ev.tok, ev.doc_start)
    doc_of = np.repeat(np.arange(ev.n_docs, dtype=np.int64), np.diff(ev.doc_start))
    out[KNMODEL] = {"bits_per_byte": bits_per_byte(lp, ev.n_bytes), "log2p": lp,
                    "doc_bits": np.bincount(doc_of, weights=-lp, minlength=ev.n_docs),
                    "bits_per_token": float(-lp.sum() / max(1, lp.shape[0]))}
    out["kn5_knmodel_bits_per_byte"] = out[KNMODEL]["bits_per_byte"]
    return out


def deoverlap(train_set: np.ndarray, ev, n: int = 13) -> tuple[np.ndarray, np.ndarray]:
    """(doc_keep bool [d], doc_frac float32 [d]) of docs/06 for the split's documents."""
    o = contamination.overlap(train_set, np.asarray(ev.tok), ev.doc_start, n)
    return o["doc_keep"].astype(bool), o["doc_frac"]


def deoverlap_keep(train_set: np.ndarray, ev, n: int = 13) -> np.ndarray:
    return deoverlap(train_set, ev, n)[0]


def summarize(prefix: str, rows: dict, doc_tokens: np.ndarray, doc_bytes: np.ndarray, keep: np.ndarray | None,
              n_boot: int = protocol.N_BOOT, label: str = "measured") -> dict:
    """Every row and every paired difference on the documents of `keep` (None: all; the full-split bits per
    byte are then the scoring pass's own numbers)."""
    sel = np.ones(doc_bytes.shape[0], dtype=bool) if keep is None else keep
    by, tk = doc_bytes[sel].astype(np.float64), doc_tokens[sel]
    out = {f"{prefix}.documents_tokens_bytes": m([int(sel.sum()), int(tk.sum()), int(by.sum())], label)}
    if int(sel.sum()) == 0:  # an empty kept set: no row and no interval exist on it (finish pass, rev 3)
        out[f"{prefix}.empty"] = m(True, label, note="no document in this set: its rows are not computed")
        return out
    idx = protocol.bootstrap_index(int(sel.sum()), n_boot=n_boot)
    for name in ALL_ROWS:
        bits = rows[name]["doc_bits"][sel]
        bpb = rows[name]["bits_per_byte"] if keep is None else float(bits.sum() / by.sum())
        lab = rows[name].get("label", label)
        out[f"{prefix}.{name}.bits_per_byte"] = m(bpb, lab, note="total -log2 P / UTF-8 bytes (protocol CP)")
        out[f"{prefix}.{name}.bits_per_token"] = m(float(bits.sum() / max(1, tk.sum())), lab)
        b = protocol.bootstrap_bpb(bits, by, idx)
        out[f"{prefix}.{name}.bits_per_byte_ci95"] = m(b["ci95"], "derived" if lab == "measured" else lab,
                                                       note=f"document bootstrap, {b['n_docs']} documents, "
                                                            f"{b['n_boot']} resamples, seed {list(protocol.BOOT_SEED)}")
    for a, b2 in PAIRS:
        r = protocol.bootstrap_bpb(rows[a]["doc_bits"][sel], by, idx, rows[b2]["doc_bits"][sel])
        # a pair that uses an unverified row is unverified too (the vote-form KN-5 row after a failed check)
        row_lab = "unverified" if "unverified" in (rows[a].get("label"), rows[b2].get("label")) else label
        lab = "derived" if row_lab == "measured" else row_lab
        out[f"{prefix}.{a}.minus_{b2}.bits_per_byte"] = m(r["point"], lab, ci95=r["ci95"], note="paired document bootstrap")
    return out


def save_scores(path: Path, rows: dict, ev, keep: np.ndarray, frac: np.ndarray) -> str:
    arrays = {"doc_start": np.asarray(ev.doc_start, dtype=np.int64), "doc_bytes": np.asarray(ev.doc_bytes, dtype=np.int64),
              "doc_tokens": np.diff(np.asarray(ev.doc_start, dtype=np.int64)), "doc_keep": keep.astype(np.uint8),
              "doc_frac": np.asarray(frac, dtype=np.float32)}
    for name in ALL_ROWS:
        arrays[f"{name}.doc_bits"] = np.asarray(rows[name]["doc_bits"], dtype=np.float64)
        arrays[f"{name}.log2p"] = np.asarray(rows[name]["log2p"], dtype=np.float64)
    with open(path, "xb") as handle:
        np.savez(handle, **arrays)
    return hashing.sha256_file(path)


def verdicts(rows: dict, cfg: dict, valid: bool) -> dict:
    """H8.3: CVTM-core test bits per byte at most the threshold; H8.4: strictly below KN-5+D0. With a
    failed invalidating check (valid False) the numbers are kept and the verdict is "invalid"."""
    core, cache = rows["cvtm_core"]["bits_per_byte"], rows["kn5_D0"]["bits_per_byte"]
    thr = float(cfg["thresholds"]["H8.3_max_test_bits_per_byte"])
    state = (lambda ok: ("met" if ok else "not met") if valid else "invalid: an invalidating check failed")
    return {"verdict.H8.3": m({"verdict": state(core <= thr), "cvtm_core_test_bits_per_byte": core, "threshold": thr},
                              "derived", rule="docs/13 line 178: at most 1.3219 (KN-5 test 1.3719 minus 0.05)"),
            "verdict.H8.4": m({"verdict": state(core < cache), "cvtm_core": core, "kn5_D0": cache},
                              "derived", rule="docs/13 line 179: lower than KN-5 plus a document unigram cache")}


def verdict_valid(failed: list[str]) -> bool:
    """Valid unless a check failed other than the two reporting-only test checks."""
    return not [f for f in failed if f not in REPORTING_ONLY]


def exact_checks(rows: dict, ref: dict, split: str) -> dict:
    """{check: (ok, details)} against the frozen references of the split (exact float equality)."""
    out = {}
    for key, got in ((f"{split}_cvtm_core", rows["cvtm_core"]["bits_per_byte"]),
                     (f"{split}_kn5_D0", rows["kn5_D0"]["bits_per_byte"]),
                     (f"{split}_kn5_knmodel", rows["kn5_knmodel_bits_per_byte"])):
        if key in ref:
            out[f"reproduces.{key}"] = (got == float(ref[key]), {"got": got, "want": ref[key]})
    if f"{split}_kn5_knmodel" in ref:
        d = abs(rows["kn5"]["bits_per_byte"] - rows["kn5_knmodel_bits_per_byte"])
        out[f"{split}.kn5_vote_form_within_1e-9_of_knmodel"] = (d <= 1e-9, {"abs_diff": d})
    return out


def heading_present(docs13: str, words: str) -> bool:
    return any(line.startswith("### A") and words in line for line in docs13.splitlines())


def run_identity_ok(cfg: dict, run_dir: str | None) -> bool:
    if not run_dir:
        return False
    try:
        rec = json.loads((Path(run_dir) / "run.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return rec.get("name") == cfg["test"]["run_name"] and rec.get("status") == "running"


def test_read_refusals(cfg: dict, args, docs13: str, rows: list[dict], log_sha: str | None, failed: list[str],
                       run_dir: str | None) -> list[str]:
    """Every reason not to open the test split, each naming its condition (empty: the read may happen)."""
    t = cfg["test"]
    why = []
    if failed:
        why.append("checks failed before the read: " + ", ".join(failed))
    if not args.confirm_test_read:
        why.append("--confirm-test-read not given")
    if not run_identity_ok(cfg, run_dir):
        why.append(f"not run by scripts/run_logged.py as {t['run_name']!r} (TMLM_RUN_DIR/run.json, status running)")
    if not heading_present(docs13, t["amendment_heading"]):
        why.append(f"docs/13 has no heading line '### A...' with {t['amendment_heading']!r}: the plan is not frozen")
    if len(rows) != int(t["access_rows_before"]):
        why.append(f"the access log has {len(rows)} rows, the plan declares {t['access_rows_before']}")
    if log_sha != t["access_log_sha256_before"]:
        why.append(f"the access log sha256 is {log_sha}, the plan pins {t['access_log_sha256_before']}")
    earlier = [i + 1 for i, r in enumerate(rows) if r.get("purpose") == t["purpose"]]
    if earlier:
        why.append(f"access-log rows {earlier} already carry this read's purpose")
    return why


def row_is_this_run(row: dict, cfg: dict, started: str) -> bool:
    return (row.get("caller") == "corpus_store.load_split" and row.get("purpose") == cfg["test"]["purpose"]
            and row.get("pid") == os.getpid() and started <= str(row.get("utc", "")) <= utc_now())


# ------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", choices=("validation", "test"), required=True)
    ap.add_argument("--confirm-test-read", action="store_true")
    ap.add_argument("--out-dir", default=None, help="output folder when not run under scripts/run_logged.py")
    args = ap.parse_args(argv)
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    run_dir = os.environ.get("TMLM_RUN_DIR")
    out = Path(run_dir or args.out_dir or "")
    if not str(out) or str(out) == ".":
        raise SystemExit("not under run_logged.py: give --out-dir (a scratch folder)")
    out.mkdir(parents=True, exist_ok=True)
    metrics: dict = {}
    failed: list[str] = []
    t0 = time.time()
    started = utc_now()
    metrics["driver"] = m({"pid": os.getpid(), "started_utc": started, "split": args.split,
                           "confirm_test_read": bool(args.confirm_test_read)})

    def save() -> None:
        metrics["wall_s"] = m(round(time.time() - t0, 1), unit="s")
        metrics["peak_rss_gib"] = m(peak_rss_gib(), note="VmHWM of the driver process")
        metrics["checks_failed"] = m(list(failed))
        if run_dir:
            from tmlm.common import runlog
            runlog.write_metrics(metrics, on_conflict="replace")
        else:
            (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=str) + "\n", encoding="utf-8")

    def check(name: str, ok: bool, details: dict) -> None:
        metrics[f"check.{name}"] = m(bool(ok), details=details)
        if not ok:
            failed.append(name)
        print(f"check {name}: {'ok' if ok else 'FAILED'} {details}", flush=True)

    # ---- before the read: code, freeze, configuration, data identity
    for name, ok, details in code_checks(cfg) + freeze_file_checks(cfg):
        check(name, ok, details)
    boot = cfg["bootstrap"]
    check("config.bootstrap_matches_protocol", int(boot["resamples"]) == protocol.N_BOOT
          and tuple(boot["seed"]) == tuple(protocol.BOOT_SEED), {"config": boot})
    splits = ("train", "validation", "test") if args.split == "test" else ("train", "validation")
    dchecks, digests = data_checks(cfg, splits)
    for name, ok, details in dchecks:
        check(name, ok, details)
    metrics["dataset_hashes"] = m(protocol.write_dataset_hashes(splits, protocol.CORPUS, int(cfg["vocab"])),
                                  note="digests from the stores' meta.json files only; no split is loaded for them")
    rows_at_start = access_rows()
    metrics["access_log_at_start"] = m({"rows": len(rows_at_start), "sha256": access_log_sha256()})
    core = load_frozen_core(cfg)
    check("freeze.core_state_hash", core.state_hash() == cfg["frozen"]["core_state_hash"], {"got": core.state_hash()})
    variants = {name: core if comps == "full" else core.variant(comps) for name, comps in ROWS}
    tok, ds = protocol.load_train(protocol.CORPUS, int(cfg["vocab"]), check=True)
    train_set = contamination.train_ngram_set(np.asarray(tok), ds, 13)
    del tok, ds
    m2 = contamination.contamination_dir(protocol.CORPUS, int(cfg["vocab"]))

    def score_split(split: str, ev) -> None:
        rows = score_rows(core, variants, ev)
        keep, frac = deoverlap(train_set, ev)
        want = json.loads((m2 / f"{split}_13gram.meta.json").read_text(encoding="utf-8"))["sha256"]["doc_keep"]
        flags_ok = hashing.sha256_array(keep.astype(np.uint8)) == want
        check(f"{split}.doc_keep_equals_M2_flags", flags_ok, {})
        exact = exact_checks(rows, cfg["reference"], split)
        for name, (ok, details) in exact.items():
            check(name, ok, details)
        if split == "test" and not exact.get("test.kn5_vote_form_within_1e-9_of_knmodel", (True, {}))[0]:
            rows["kn5"]["label"] = "unverified"
        doc_tokens = np.diff(np.asarray(ev.doc_start, dtype=np.int64))
        metrics.update(summarize(split, rows, doc_tokens, np.asarray(ev.doc_bytes), None))
        for name in ALL_ROWS:
            print(f"{split} {name}: {rows[name]['bits_per_byte']!r} bits per byte", flush=True)
        if split == "test":
            metrics.update(verdicts(rows, cfg, verdict_valid(failed)))
            for key in ("verdict.H8.3", "verdict.H8.4"):
                print(f"{key}: {metrics[key]['value']}", flush=True)
        save()  # the full-split rows and the verdicts are on disk before the de-overlapped rows and the npz
        metrics.update(summarize(f"{split}.deoverlapped", rows, doc_tokens, np.asarray(ev.doc_bytes), keep,
                                 label="measured" if flags_ok else "unverified"))
        metrics[f"{split}.scores_npz_sha256"] = m(save_scores(out / f"scores_{split}.npz", rows, ev, keep, frac),
                                                  file=f"scores_{split}.npz")

    # ---- the validation pass (always; a failed check here refuses the test read)
    ev = protocol.load_eval_split("validation", protocol.CORPUS, int(cfg["vocab"]), check=True)
    check("validation.raw_doc_bytes_sha256", raw_doc_bytes_sha256("validation") == cfg["frozen"]["data_digests"]["validation_doc_bytes"], {})
    score_split("validation", ev)
    del ev
    if args.split == "validation":
        check("validation.no_test_access", len(access_rows()) == len(rows_at_start), {"rows": len(access_rows())})
        save()
        return EXIT_CHECK_FAILED if failed else EXIT_OK

    # ---- the test read: refusals first (no row is written when refused)
    docs13 = PREREG.read_text(encoding="utf-8")
    why = test_read_refusals(cfg, args, docs13, rows_at_start, access_log_sha256(), list(failed), run_dir)
    if why:
        metrics["test.refused"] = m(why)
        check("test.access_log_unchanged_after_refusal", len(access_rows()) == len(rows_at_start), {})
        print("TEST READ REFUSED: " + "; ".join(why), flush=True)
        save()
        return EXIT_REFUSED
    try:  # from here on the split counts as used once its row is written: a crash keeps what exists (exit 4)
        ev = protocol.load_eval_split("test", protocol.CORPUS, int(cfg["vocab"]), purpose=cfg["test"]["purpose"],
                                      check=True)
        rows_after = access_rows()
        metrics["test.access_log_after"] = m({"rows": len(rows_after), "sha256": access_log_sha256()})
        metrics["test.access_row"] = m(rows_after[-1] if rows_after else None)
        check("test.one_access_row_appended", len(rows_after) == len(rows_at_start) + 1, {"rows": len(rows_after)})
        check("test.access_row_is_this_run", bool(rows_after) and row_is_this_run(rows_after[-1], cfg, started),
              {"pid": os.getpid()})
        check("test.raw_doc_bytes_sha256",
              raw_doc_bytes_sha256("test") == cfg["frozen"]["data_digests"]["test_doc_bytes"], {})
        save()  # the access-row checks are on disk before the scoring
        score_split("test", ev)
    except Exception as e:  # noqa: BLE001 (any crash after the read: record it, keep the numbers, no re-read)
        rows_now = access_rows()
        metrics["test.crashed_after_read"] = m({"error": f"{type(e).__name__}: {e}", "access_rows_now": len(rows_now),
                                                "row_appended": len(rows_now) > len(rows_at_start)},
                                               note="the numbers written so far are kept; the split is not read again")
        for key in ("verdict.H8.3", "verdict.H8.4"):
            value = dict(metrics[key]["value"]) if key in metrics else {}
            value["verdict"] = "invalid: crashed after the read"
            metrics[key] = m(value, "derived", rule="exit 4: a crash after the read leaves both verdicts invalid")
        failed.append("test.crashed_after_read")
        print(f"CRASHED AFTER THE READ: {type(e).__name__}: {e}", flush=True)
        save()
        return EXIT_CRASHED_AFTER_READ
    save()
    return EXIT_CHECK_FAILED if failed else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
