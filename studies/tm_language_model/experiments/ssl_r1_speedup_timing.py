"""Measurement only (config experiments/cvtm/configs/ssl_r1_speedup_timing.json): the wall time of the registered
prefix work (ssl_r1_pretext.fire_counts in the features stage; ssl_r1_controls.count_table and fit_rows_fired in the
Q_pred stage) against the one pass tmlm.ssl.qpred_fast.prefix_pass (speedup O10), interleaved in one process on the
same checkpoint state. No pass rule: the correctness of O10 is decided by experiments/ssl_r1_speedup_verify.py.
Reads only the training prefix (no validation or test data)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
for _q in (ROOT / "src", EXPERIMENTS):
    if str(_q) not in sys.path:
        sys.path.insert(0, str(_q))

import ssl_r1_controls as C  # noqa: E402  (imported, never edited)
import ssl_r1_pretext as R  # noqa: E402  (imported, never edited)
from ssl_r1_speedup_verify import Rec  # noqa: E402
from tmlm.common import hashing  # noqa: E402
from tmlm.ssl import pretext as P  # noqa: E402
from tmlm.ssl import qpred_fast as QF  # noqa: E402

T0 = time.time()
DEFAULT_CONFIG = ROOT / "experiments" / "cvtm" / "configs" / "ssl_r1_speedup_timing.json"


def main(argv: list[str] | None = None) -> int:
    import cupy as cp

    import cvtm_t1_readout as T1

    ap = argparse.ArgumentParser(description="prefix-pass timing for O10 (measurement only)")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = ap.parse_args(argv)
    tcfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    ctl = C.load_config(ROOT / tcfg["controls_config"])
    p = ctl["params"]
    rec = Rec(False)
    rec.m("stage", "passtiming")
    rec.m("config_params", p)
    rec.m("timing.config_sha256", hashing.sha256_file(Path(args.config)))
    C.check_shared_params(ctl, rec)
    tr = R.rp(tcfg["references"]["train_run"])
    arm, seed = C.train_run_identity(tr)
    base, _ = C.CONTROL_ARMS[arm]
    prefix = R.load_prefix(p, rec)
    ad = P.arm_data(base, prefix, int(p["vocab"]), float(p["f1_t"]))
    prefix_t = P.with_tables(prefix, kind_map=ad.kind_map, sym_freq=ad.sym_freq)
    _, fit_pos, cnt_pos = C.qpred_split(prefix)
    tok = np.asarray(prefix.tok).astype(np.int64)
    n = int(prefix.tok.shape[0])
    zero = np.zeros(n, np.uint8)
    chunk = int(p["forward_chunk"])
    V = int(p["vocab"])
    check_cnt = T1.sample_positions(np.arange(cnt_pos.shape[0]), int(p["qpred"]["checks"]["cross_engine_count_rows"]), 3)
    for name in tcfg["checkpoints"]:
        cfg, st, _ = C.load_named_checkpoint(tr, name, arm, seed, p, rec)
        Cn = int(cfg.n_clauses)
        eng = R.make_engine("cuda", cfg, prefix_t)
        eng.load_state(st)
        hidden = R.hidden_for(cfg, n)
        reps, same = [], None
        for r in range(int(tcfg["repeats"])):
            t0 = time.time()
            fc = R.fire_counts(eng, cfg, n, chunk)
            t_fc = time.time() - t0
            t0 = time.time()
            counts, n_ev, got = C.count_table(eng, "cuda", cnt_pos, tok[cnt_pos], Cn, V, chunk, zero, check_cnt)
            t_ct = time.time() - t0
            t0 = time.time()
            fptr, fids = C.fit_rows_fired(eng, "cuda", fit_pos, zero, chunk)
            t_fr = time.time() - t0
            t0 = time.time()
            pp = QF.prefix_pass(eng, Cn, V, n, chunk, hidden, tok, cnt_pos, fit_pos, check_cnt,
                                lambda pos: C.fit_rows_fired(eng, "cuda", pos, zero, chunk))
            t_pp = time.time() - t0
            if same is None:
                lg = pp.got()
                same = {"fire_counts": bool(np.array_equal(pp.fire_counts, fc)), "counts": bool(np.array_equal(pp.counts_dense(), counts)),
                        "events": int(pp.n_events) == int(n_ev), "fit_rows": bool(np.array_equal(pp.fit_ptr, fptr) and np.array_equal(pp.fit_ids, fids)),
                        "check_rows": sorted(lg) == sorted(got) and all(np.array_equal(lg[k], got[k]) for k in got)}
            reps.append({"fire_counts_s": t_fc, "count_table_s": t_ct, "fit_rows_s": t_fr, "one_pass_s": t_pp})
            rec.log(f"{name} repeat {r}: fire_counts {t_fc:.1f} s, count_table {t_ct:.1f} s, fit rows {t_fr:.1f} s, one pass {t_pp:.1f} s")
            del counts, pp, fc
            cp.get_default_memory_pool().free_all_blocks()
            rec.flush()
        med = {k: float(np.median([x[k] for x in reps])) for k in reps[0]}
        registered = med["fire_counts_s"] + med["count_table_s"] + med["fit_rows_s"]
        rec.m(f"timing.{name}", {"repeats": reps, "median": med, "registered_total_s": registered, "one_pass_s": med["one_pass_s"],
                                 "saving_s": registered - med["one_pass_s"],
                                 "one_pass_minus_fire_counts_s": med["one_pass_s"] - med["fire_counts_s"],
                                 "outputs_equal_first_repeat": same},
              note="interleaved in one process on a GPU shared with other jobs (shared lock); medians over the repeats")
        del eng
        cp.get_default_memory_pool().free_all_blocks()
    rec.m("wall_total_s", time.time() - T0)
    rec.m("failed_checks", rec.failed)
    rec.flush()
    rec.log(f"passtiming done; failed checks: {rec.failed}")
    return 1 if rec.failed else 0


if __name__ == "__main__":
    sys.exit(main())
