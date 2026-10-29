"""Independent recompute of a run of experiments/cvtm_test_read.py from its own output files (the report step
of the plan of the H8.3 and H8.4 test read). It opens no data split: it reads the run folder's metrics.json and
scores_<split>.npz only, so it adds no access row.

For one split it recomputes from the per-document and per-token arrays: every row's bits per byte on the full
split and on the de-overlapped documents (equal to the reported numbers within 1e-12), the per-document bits
against the per-token log2 P (within 1e-9 bits), the paired differences with a NEW bootstrap seed (reported
beside the run's intervals; they are not expected to be equal) on the full split and on the de-overlapped
documents, and, for the test split, both verdicts from the recomputed numbers (equal to the written ones), the
KN-5 identity (the KNModel row equals the CP anchor's 1.3718501779650507 within 1e-12) and the split's size
(322,929 tokens, 1,289,123 bytes). Exit status 0 when everything agrees, 1 otherwise. What it does not redo: the
scoring itself, which rests on the exact reproduction of every frozen validation number and on the KN-5 identity.

    .venv-tm/bin/python experiments/cvtm_test_read_verify.py --run-dir results/<UTC>_cvtm_a3_H8_3_H8_4_test --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tmlm.common import hashing  # noqa: E402

ROWS = ("cvtm_core", "kn5_D0", "kn5", "kn5_knmodel")
PAIRS = (("cvtm_core", "kn5_D0"), ("cvtm_core", "kn5"), ("kn5_D0", "kn5"), ("cvtm_core", "kn5_knmodel"))
NEW_SEED = (20261025, 113, 2)
N_BOOT = 2000
KN5_TEST = 1.3718501779650507              # CP anchor, kn5.test.bits_per_byte
TEST_TOKENS, TEST_BYTES = 322929, 1289123      # CP anchor metrics


def boot(diff: np.ndarray, by: np.ndarray, seed=NEW_SEED, n_boot: int = N_BOOT) -> list[float]:
    rng = np.random.default_rng(np.random.SeedSequence(list(seed)))
    idx = rng.integers(0, diff.shape[0], size=(n_boot, diff.shape[0]))
    stats = diff[idx].sum(axis=1) / by[idx].sum(axis=1)
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def verify(run_dir: Path, split: str, threshold: float = 1.3219, kn5_test: float = KN5_TEST,
           test_tokens: int = TEST_TOKENS, test_bytes: int = TEST_BYTES) -> dict:
    met = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    npz = run_dir / f"scores_{split}.npz"
    out = {"npz_sha256_equal": hashing.sha256_file(npz) == met[f"{split}.scores_npz_sha256"]["value"]}
    with np.load(npz) as z:
        a = {k: z[k] for k in z.files}
    by = a["doc_bytes"].astype(np.float64)
    keep = a["doc_keep"].astype(bool)
    doc_of = np.repeat(np.arange(by.shape[0]), a["doc_tokens"])
    worst, tok_vs_doc = 0.0, 0.0
    for row in ROWS:
        bits = a[f"{row}.doc_bits"]
        tok_vs_doc = max(tok_vs_doc, float(np.max(np.abs(np.bincount(doc_of, weights=-a[f"{row}.log2p"],
                                                                         minlength=by.shape[0]) - bits))))
        sets = ((split, np.ones_like(keep)),) + (((f"{split}.deoverlapped", keep),) if keep.any() else ())
        for prefix, sel in sets:  # an empty kept set has no rows (the driver writes <split>.deoverlapped.empty)
            got = float(bits[sel].sum() / by[sel].sum())
            worst = max(worst, abs(got - met[f"{prefix}.{row}.bits_per_byte"]["value"]))
    out["max_abs_diff_bits_per_byte"] = worst
    out["max_abs_diff_doc_bits_vs_tokens"] = tok_vs_doc
    out["pairs_new_seed"] = {f"{x}.minus_{y}": {"new_ci95": boot(a[f"{x}.doc_bits"] - a[f"{y}.doc_bits"], by),
                                                "run_ci95": met[f"{split}.{x}.minus_{y}.bits_per_byte"]["ci95"]}
                             for x, y in PAIRS}
    if keep.any():
        out["pairs_new_seed_deoverlapped"] = {
            f"{x}.minus_{y}": {"new_ci95": boot(a[f"{x}.doc_bits"][keep] - a[f"{y}.doc_bits"][keep], by[keep]),
                               "run_ci95": met[f"{split}.deoverlapped.{x}.minus_{y}.bits_per_byte"]["ci95"]}
            for x, y in PAIRS}
    ok = out["npz_sha256_equal"] and worst <= 1e-12 and tok_vs_doc <= 1e-9
    if split == "test":
        core = float(a["cvtm_core.doc_bits"].sum() / by.sum())
        cache = float(a["kn5_D0.doc_bits"].sum() / by.sum())
        h83 = met["verdict.H8.3"]["value"]["verdict"]
        h84 = met["verdict.H8.4"]["value"]["verdict"]
        out["recomputed_verdicts"] = {"H8.3": core <= threshold, "H8.4": core < cache}
        kn = float(a["kn5_knmodel.doc_bits"].sum() / by.sum())
        out["kn5_identity"] = {"recomputed": kn, "registered": kn5_test, "equal_within_1e-12": abs(kn - kn5_test) <= 1e-12}
        n_tok, n_byte = int(a["doc_tokens"].sum()), int(by.sum())
        out["split_size"] = {"tokens": n_tok, "bytes": n_byte, "expected": [test_tokens, test_bytes],
                             "equal": n_tok == test_tokens and n_byte == test_bytes}
        ok = ok and out["kn5_identity"]["equal_within_1e-12"] and out["split_size"]["equal"]
        if not h83.startswith("invalid"):
            ok = ok and (h83 == "met") == (core <= threshold) and (h84 == "met") == (core < cache)
    out["agrees"] = bool(ok)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--split", choices=("validation", "test"), required=True)
    ap.add_argument("--json", default=None, help="write the report here")
    args = ap.parse_args(argv)
    rep = verify(Path(args.run_dir), args.split)
    text = json.dumps(rep, indent=1)
    print(text)
    if args.json:
        Path(args.json).write_text(text + "\n", encoding="utf-8")
    return 0 if rep["agrees"] else 1


if __name__ == "__main__":
    sys.exit(main())
