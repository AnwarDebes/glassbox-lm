#!/usr/bin/env python
"""Test 5, the registered TM-AE run (REGISTRATION_T5.md, "Fixed", TM-AE).

    python -I train_tmae.py            # the registered run (pilot estimate: 3.7-7.5 h on one CPU thread)
    python -I train_tmae.py --tiny     # 2 rounds on all targets (< 60 s), *_tiny files only

tmu 0.8.3 TMAutoEncoder through tmae_fixed.MaskedClassSumAE (the paper's masked class sum, Eq. 4 of arXiv
2301.00709): 600 clauses, T 1200, s 5.0, accumulation 25, feature_negation False, output_balancing 0.5. All 4,564
corpus-(a) tokens (t5/corpus_a.npz) are both targets and features. 2,000 rounds as 200 fit calls of 10 rounds
(one round = one example per target). Seed 1; before every fit call libc srand() with a value derived from
(seed, call index) (tmu draws the documents with unseeded libc rand()). CPU only (CUDA_VISIBLE_DEVICES=""), one
thread.

Outputs (all in t5/):
- tmae_ckpt_r{rounds}.npz every 250 rounds: W (targets x clauses, int32), include (clauses x literals, bool; the
  negated half is never included since feature_negation is off), vocab_ids (potion token ids of the targets, in
  corpus_a.npz order), targets, rounds, T, n_clauses, config (JSON). Written atomically (temp file + rename).
- runs_tmae.log: after every fit call one line with a `date -u` timestamp, rounds done and seconds per update.
- tmae_done.json at the end: rounds done, the reason (completed or budget), the last checkpoint and its sha256,
  train time, weight histogram of the final W.
Budget: the run stops cleanly after the fit call that crosses 10 h of training; the last checkpoint is kept and
named in tmae_done.json (no partial checkpoint is written, no resume).
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # before tmu: tmu.util.cuda_profiler runs `import pycuda.autoinit`
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import ctypes  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import resource  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.sparse import csr_matrix  # noqa: E402

T5 = Path(__file__).resolve().parent
sys.path.insert(0, str(T5))  # python -I leaves the script folder off sys.path; tmae_fixed.py lives here
CORPUS, CORPUS_MAN = T5 / "corpus_a.npz", T5 / "corpus_a.manifest.json"
LIBC = ctypes.CDLL("libc.so.6")  # the libc whose rand() tmu's Tools.c uses for the document draws

CFG = dict(n_clauses=600, T=1200, s=5.0, accumulation=25, feature_negation=False, output_balancing=0.5,
           number_of_state_bits_ta=8, rounds=2000, rounds_per_call=10, ckpt_every=250, seed=1, budget_h=10.0,
           cls="tmae_fixed.MaskedClassSumAE", platform="CPU")
TINY = dict(rounds=2, rounds_per_call=1, ckpt_every=1)


def utc() -> str:
    try:
        return subprocess.run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # pragma: no cover - date is always there; keep the run alive if it is not
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def seed_libc(seed: int, call: int) -> int:
    v = (seed * 1_000_003 + call * 7_919 + 12_345) % (2**32)
    LIBC.srand(ctypes.c_uint(v))
    return v


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rss_mb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)


def weight_hist(W: np.ndarray) -> dict:
    labels = ["<=-100", "-99..-10", "-9..-2", "-1", "0", "1", "2..9", "10..99", ">=100"]
    edges = [-np.inf, -99.5, -9.5, -1.5, -0.5, 0.5, 1.5, 9.5, 99.5, np.inf]
    counts, _ = np.histogram(W, bins=edges)
    return dict(zip(labels, (int(c) for c in counts)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true", help="2 rounds on all targets, *_tiny files only")
    args = ap.parse_args()
    cfg = dict(CFG, **TINY) if args.tiny else dict(CFG)
    sfx = "_tiny" if args.tiny else ""
    log_path, done_path = T5 / f"runs_tmae{sfx}.log", T5 / f"tmae_done{sfx}.json"

    def log(msg: str) -> None:
        with open(log_path, "a") as f:
            f.write(f"{utc()} {msg}\n")

    import tmu  # noqa: F401  (after the environment settings above)
    from importlib.metadata import version

    from tmae_fixed import MaskedClassSumAE

    z = np.load(CORPUS)
    n_docs, F = (int(v) for v in z["shape"])
    X = csr_matrix((np.ones(len(z["indices"]), dtype=np.uint32), z["indices"], z["indptr"]), shape=(n_docs, F))
    assert X.indices.dtype == np.int32 and X.indptr.dtype == np.int32
    vocab_ids = z["vocab_ids"].astype(np.int32)
    corpus_sha = json.loads(CORPUS_MAN.read_text())["outputs"]["npz_sha256"]
    assert corpus_sha == sha256_file(CORPUS), "corpus_a.npz differs from its manifest"
    targets = np.arange(F, dtype=np.uint32)
    seed = cfg["seed"]
    assert seed % 2 == 1, "odd seeds only: tmu's pcg32 is multiplicative"
    tm = MaskedClassSumAE(number_of_clauses=cfg["n_clauses"], T=cfg["T"], s=cfg["s"], output_active=targets,
                          accumulation=cfg["accumulation"], feature_negation=cfg["feature_negation"],
                          platform=cfg["platform"], output_balancing=cfg["output_balancing"],
                          number_of_state_bits_ta=cfg["number_of_state_bits_ta"], seed=seed)
    script_sha = sha256_file(Path(__file__).resolve())[:16]
    fixed_sha = sha256_file(T5 / "tmae_fixed.py")[:16]
    log(f"start{' TINY' if args.tiny else ''}: docs {n_docs}, F {F} (targets = features), corpus {corpus_sha[:16]}, "
        f"train_tmae.py {script_sha}, tmae_fixed.py {fixed_sha}, tmu {version('tmu')}, numpy {np.__version__}, "
        f"python {platform.python_version()}, config {json.dumps(cfg)}")

    budget_s = cfg["budget_h"] * 3600.0
    t_start = time.perf_counter()
    c_start = time.process_time()
    rounds_done, call, last_ckpt, reason = 0, 0, None, "completed"
    walls = []
    while rounds_done < cfg["rounds"]:
        r = min(cfg["rounds_per_call"], cfg["rounds"] - rounds_done)
        sv = seed_libc(seed, call)
        w0, c0 = time.perf_counter(), time.process_time()
        tm.fit(X, number_of_examples=r)
        dw, dc = time.perf_counter() - w0, time.process_time() - c0
        rounds_done += r
        call += 1
        walls.append((dw, r))
        elapsed = time.perf_counter() - t_start
        log(f"call {call} rounds {rounds_done}/{cfg['rounds']} s_per_update {dw / (r * F):.6f} "
            f"call_wall_s {dw:.1f} call_cpu_s {dc:.1f} elapsed_h {elapsed / 3600:.3f} srand {sv} rss_peak_mb {rss_mb()}")
        if rounds_done % cfg["ckpt_every"] == 0:
            W = np.stack([tm.get_weights(i).copy() for i in range(F)]).astype(np.int32)
            include = tm.clause_bank.get_literals().astype(bool)
            p = T5 / f"tmae_ckpt{sfx}_r{rounds_done}.npz"
            tmp = p.with_name(p.stem + ".partial.npz")
            np.savez(tmp, W=W, include=include, vocab_ids=vocab_ids, targets=targets, rounds=rounds_done,
                     T=cfg["T"], n_clauses=cfg["n_clauses"], config=json.dumps(cfg))
            os.replace(tmp, p)
            last_ckpt = p
            log(f"checkpoint {p.name} sha256 {sha256_file(p)[:16]} (W {W.shape}, include {include.shape})")
        if elapsed > budget_s and rounds_done < cfg["rounds"]:
            reason = "budget"
            break
    train_wall, train_cpu = time.perf_counter() - t_start, time.process_time() - c_start
    W = np.stack([tm.get_weights(i).copy() for i in range(F)]).astype(np.int32)
    done = {
        "utc_end": utc(), "tiny": args.tiny, "registration": "REGISTRATION_T5.md", "reason": reason,
        "rounds_done": rounds_done, "rounds_planned": cfg["rounds"], "fit_calls": call,
        "last_checkpoint": last_ckpt.name if last_ckpt else None,
        "last_checkpoint_sha256": sha256_file(last_ckpt) if last_ckpt else None,
        "last_checkpoint_rounds": int(np.load(last_ckpt)["rounds"]) if last_ckpt else None,
        "train_wall_s": round(train_wall, 1), "train_cpu_s": round(train_cpu, 1),
        "s_per_update_mean": train_wall / max(1, rounds_done * F),
        "s_per_update_last_call": walls[-1][0] / (walls[-1][1] * F) if walls else None,
        "weight_hist_final": weight_hist(W), "rss_peak_mb": rss_mb(), "config": cfg,
        "corpus_sha256": corpus_sha, "train_tmae_sha256": script_sha, "tmae_fixed_sha256": fixed_sha,
    }
    done_path.write_text(json.dumps(done, indent=1))
    log(f"end: {reason}, rounds {rounds_done}, last checkpoint {done['last_checkpoint']}, "
        f"train {train_wall / 3600:.3f} h wall, {done['s_per_update_mean']:.6f} s per update")
    print(json.dumps({k: v for k, v in done.items() if k != "config"}))


if __name__ == "__main__":
    main()
