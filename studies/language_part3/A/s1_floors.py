#!/usr/bin/env python
"""S1 (CPU): the 10 conditional counting floors on fit, chosen on tune-cal; tune-hold reported; KenLM check K1.

    chain/part3_cpu_job.sh A tm s1_floors.py [--s0 DIR] [--out DIR]

Order of the picks (all on tune-cal bits per token): EM weights of F2 n3 / n5 on tune-cal; then the better F2 and
the better F3 context; then F4 (on that F3) and F5 (on that F2); then the better F4; then F6's EM weights (that F2
and that F4); the floor is the configuration with the lowest tune-cal bits. tune-hold bits of all 10 and of the
unconditional KN-5 reference are reported unchanged.
K1: tally KN-5 (modified KN, three discounts per order, each estimated from the counts: asserted) against KenLM
lmplz -o 5 (-S 4G, -T in A/tmp) on the fit text, scored on tune-hold: |PPL_own / PPL_kenlm - 1| <= 1e-3, else STOP
(exit code 3 after the outputs are written).
K2 (floor part), before any number is logged: at 1,000 random tune positions (seed 0) the full vector of every one of
the 10 configurations and of the KN-5 reference sums to 1 within 1e-6, is > 0, and gives the scored bits of the true
token (within 1e-9 bits); the chosen floor's vectors at these positions are kept for every arm's K2.
Refuses to start unless LOG.md holds the SHA-256 of S0's bit_list.json (frozen before the first tune-cal score).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from astruct import paths
from astruct.data import CAL, FIT, HOLD, Corpus
from astruct.floors import CONFIGS, FloorSet, bits_true, summarize
from astruct.gates import k1
from astruct.positions import Positions

paths.add_study_paths(tally=True)
from tally.floor import kenlm as tk

N_CONTRACT = 1000


def lmplz(order: int, text: Path, arpa: Path, tmp: Path, memory: str = "4G", fallback: bool = False) -> dict:
    """tally kenlm.lmplz's call (-o, -S, -T, --arpa), plus --discount_fallback for the synthetic dry run only."""
    import re
    import subprocess
    cmd = [str(paths.KENLM_BIN / "lmplz"), "-o", str(order), "-S", memory, "-T", str(tmp), "--arpa", str(arpa)]
    if fallback:
        cmd.append("--discount_fallback")
    with open(text) as fi:
        r = subprocess.run(cmd, stdin=fi, capture_output=True, text=True, check=True)
    Path(str(arpa) + ".log").write_text(r.stderr)
    return {int(m.group(1)): [float(m.group(i)) for i in (3, 4, 5)]
            for m in re.finditer(r"^(\d+) (\d+) D1=(\S+) D2=(\S+) D3\+=(\S+)$", r.stderr, re.MULTILINE)}


def kenlm_check(c: Corpus, fit_stream, hold_stream, own_kn5, d: Path, log, fallback: bool = False) -> dict:
    d.mkdir(parents=True, exist_ok=True)
    sort_tmp = paths.TMP / "kenlm_sort"
    sort_tmp.mkdir(parents=True, exist_ok=True)
    tk.write_text(fit_stream, c.vocab, c.END, str(d / "fit.txt"))
    tk.write_text(hold_stream, c.vocab, c.END, str(d / "hold.txt"))
    t = time.time()
    disc = lmplz(5, d / "fit.txt", d / "kn5.arpa", sort_tmp, memory="4G", fallback=fallback)
    tk.build_binary(str(d / "kn5.arpa"), str(d / "kn5.bin"))
    lp_k, oov = tk.query(str(d / "kn5.bin"), str(d / "hold.txt"))
    lp_o = own_kn5.logprob(hold_stream)
    if len(lp_k) != len(lp_o):
        raise SystemExit(f"KenLM scored {len(lp_k)} tokens, own {len(lp_o)}")
    rel = float(np.exp(lp_k.mean() - lp_o.mean()) - 1)
    res = {"discounts_kenlm": disc, "discounts_own": own_kn5.summary()["discounts"], "oov_tokens": int(oov.sum()),
           "tokens": len(lp_o), "ppl_own": float(np.exp(-lp_o.mean())), "ppl_kenlm": float(np.exp(-lp_k.mean())),
           "max_abs_logp_diff": float(np.abs(lp_k - lp_o).max()), "seconds": round(time.time() - t, 1), **k1(rel)}
    log("K1", json.dumps(res))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s0", default=str(paths.OUT / "s0"))
    ap.add_argument("--out", default=str(paths.OUT / "s1"))
    ap.add_argument("--logmd", default=str(paths.A_DIR.parent / "LOG.md"))
    ap.add_argument("--kenlm-discount-fallback", action="store_true",
                    help="synthetic dry run only: tiny corpora give no valid KN discounts")
    args = ap.parse_args()
    env = paths.check_env()
    if args.kenlm_discount_fallback and paths.is_real_output(Path(args.out)):
        raise SystemExit("--kenlm-discount-fallback is for the synthetic dry run only")
    bl = paths.sha256_file(Path(args.s0) / "bit_list.json")
    if bl != json.loads((Path(args.s0) / "done.json").read_text())["bit_list_sha256"]:
        raise SystemExit("bit_list.json changed after S0")
    if bl not in Path(args.logmd).read_text():
        raise SystemExit(f"LOG.md has no line with the bit-list SHA-256 {bl}; log it before the first tune-cal score")
    out = paths.OutDir(Path(args.out))
    log = paths.Log(out / "log.txt")
    t0 = time.time()
    c = Corpus.load(Path(args.s0))
    fit, cal, hold = c.replies(FIT), c.replies(CAL), c.replies(HOLD)
    log("S1 start; replies fit / tune-cal / tune-hold", len(fit), len(cal), len(hold), "| V", c.V)
    fs = FloorSet(c, fit, picks={}, need=CONFIGS, log=log)
    Pc, Ph = Positions(c, cal), Positions(c, hold)
    vc, vh = fs.view(Pc), fs.view(Ph)
    log("positions tune-cal", Pc.n, "tune-hold", Ph.n, "| fit counting positions", fs.n_count_positions)
    em = {n: fs.fit_em(vc, n) for n in ("F2n3", "F2n5")}
    # every number below stays unlogged until K2 has passed for all 10 configurations and the KN-5 reference
    cal_bits = bits_true(fs, vc, ["F1n4", "F1n6", "F2n3", "F2n5", "F3a", "F3b"])
    m = {k: float(v.mean()) for k, v in cal_bits.items()}
    fs.picks["F2"] = min(("F2n3", "F2n5"), key=lambda k: (m[k], k))
    fs.picks["F3"] = min(("F3a", "F3b"), key=lambda k: (m[k], k))
    cal_bits.update(bits_true(fs, vc, ["F4e2", "F4e4", "F5"]))
    m = {k: float(v.mean()) for k, v in cal_bits.items()}
    fs.picks["F4"] = min(("F4e2", "F4e4"), key=lambda k: (m[k], k))
    em["F6"] = fs.fit_em(vc, "F6")
    cal_bits.update(bits_true(fs, vc, ["F6"]))
    cal_bits.update(bits_true(fs, vc, ["KN5"]))
    m = {k: float(v.mean()) for k, v in cal_bits.items()}
    floor = min(CONFIGS, key=lambda k: (m[k], k))
    hold_bits = bits_true(fs, vh, list(CONFIGS) + ["KN5"])
    mh = {k: float(v.mean()) for k, v in hold_bits.items()}
    # K2 (floor part): 1,000 random tune positions (seed 0, tune-cal and tune-hold together); every configuration's
    # full vector sums to 1 within 1e-6, is > 0 everywhere, and gives the scored bits of the true token
    rng = np.random.default_rng(0)
    pick = rng.choice(Pc.n + Ph.n, size=min(N_CONTRACT, Pc.n + Ph.n), replace=False)
    keep_c = np.zeros(Pc.n, bool)
    keep_h = np.zeros(Ph.n, bool)
    keep_c[pick[pick < Pc.n]] = True
    keep_h[pick[pick >= Pc.n] - Pc.n] = True
    names = list(CONFIGS) + ["KN5"]
    k2 = {}
    for nm, v, keep, bt in (("tune-cal", vc, keep_c, cal_bits), ("tune-hold", vh, keep_h, hold_bits)):
        idx = np.flatnonzero(keep)
        if not len(idx):
            continue
        memo = fs.vectors(v, idx, names)
        for name in names:
            fv = memo[name]
            err, pmin = float(np.abs(fv.sum(1) - 1).max()), float(fv.min())
            fast = float(np.abs(-np.log2(fv[np.arange(len(idx)), v.P.x[idx]]) - bt[name][idx]).max())
            k2[f"{nm}/{name}"] = {"positions": len(idx), "max_sum_err": err, "min_p": pmin, "max_fast_err_bits": fast,
                                  "pass": bool(err <= 1e-6 and pmin > 0 and fast <= 1e-9)}
    bad = [k for k, r in k2.items() if not r["pass"]]
    if bad:
        raise SystemExit(f"K2 (floor) failed before any number was logged: {bad}")
    log(f"K2 floor: {len(k2)} checks (10 configurations and KN-5 on the tune-cal and tune-hold parts of {len(pick)} "
        f"positions) pass; max sum error {max(r['max_sum_err'] for r in k2.values()):.2e}")
    if not args.kenlm_discount_fallback:
        # K1: the in-house KN-5 is modified KN with three discounts per order, each estimated from the counts
        kn5 = fs.kn[5]
        ok = kn5.fallback_orders == [] and all(len(kn5.D[k]) == 4 and 0 < kn5.D[k][1] < 1 and 0 < kn5.D[k][2] < 2
                                               and 0 < kn5.D[k][3] < 3 for k in range(1, 6))
        if not ok:
            raise SystemExit("K1: the in-house KN-5 is not modified KN with three count-estimated discounts per order: "
                             f"{kn5.summary()['discounts']}, fallback orders {kn5.fallback_orders}")
    log("picks", json.dumps(fs.picks), "| floor", floor, "| tune-cal bits", json.dumps({k: round(v, 5) for k, v in m.items()}))
    log("tune-hold bits (reported, no choice)", json.dumps({k: round(v, 5) for k, v in mh.items()}))
    k1res = kenlm_check(c, Positions(c, fit).stream, Ph.stream, fs.kn[5], out.part / "kenlm", log,
                        fallback=args.kenlm_discount_fallback)
    k1res["kenlm_discount_fallback"] = bool(args.kenlm_discount_fallback)
    # summaries of the floor on tune-cal and tune-hold, with the full vectors at the K2 positions (every arm's K2)
    sc = summarize(fs, vc, floor, keep_full=keep_c)
    sh = summarize(fs, vh, floor, keep_full=keep_h)
    for nm, s, bt in (("tune-cal", sc, cal_bits[floor]), ("tune-hold", sh, hold_bits[floor])):
        if not np.allclose(-np.log2(s["p_true"]), bt, rtol=0, atol=1e-12):
            raise SystemExit(f"{nm}: summary bits differ from the scoring pass")
        if s["full"]:
            kp = np.array(sorted(s["full"]), np.int64)
            fv = np.stack([s["full"][k] for k in kp])
            ref = fs.vectors(vc if nm == "tune-cal" else vh, kp, [floor])[floor]
            if not np.allclose(fv, ref, rtol=0, atol=1e-12):
                raise SystemExit(f"{nm}: the kept floor vectors differ from the K2 vectors")
    for nm, s in (("cal", sc), ("hold", sh)):
        np.savez_compressed(out / f"summ_{nm}.npz", **{k: v for k, v in s.items() if k != "full"},
                            full_pos=np.array(sorted(s["full"]), np.int64),
                            full_vec=np.stack([s["full"][k] for k in sorted(s["full"])]) if s["full"] else np.zeros((0, c.V)))
    np.savez_compressed(out / "bits_cal.npz", **cal_bits)
    np.savez_compressed(out / "bits_hold.npz", **hold_bits)
    res = {"floor": floor, "picks": fs.picks, "em": fs.em_info, "lam": {k: v.tolist() for k, v in fs.lam.items()},
           "tune_cal_bits": m, "tune_hold_bits": mh, "positions": {"tune-cal": Pc.n, "tune-hold": Ph.n, "fit": fs.n_count_positions},
           "kn": {n: fs.kn[n].summary() for n in fs.kn},
           "chains": {k: [{"n_ctx": lv.n_ctx, "types": len(lv.w), "D": [float(x) for x in lv.D[1:]], "fallback": lv.fallback}
                          for lv in v] for k, v in fs.chains.items()},
           "K1": k1res, "K2": k2, "seconds": round(time.time() - t0, 1), "env": env, "code_sha256": paths.code_hashes(),
           "s0_done_sha256": paths.sha256_file(Path(args.s0) / "done.json")}
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float))
    word = "K1 PASS" if k1res["pass"] else "K1 STOP"
    log(f"floor {floor}: tune-cal {m[floor]:.5f}, tune-hold {mh[floor]:.5f} bits per token; KN-5 reference tune-hold "
        f"{mh['KN5']:.5f}; {word} (rel ppl {k1res['rel_ppl']:.2e})")
    out.done({"floor": floor, "K1": k1res["pass"], "utc_end": paths.utc()})
    if not k1res["pass"]:
        sys.exit(3)


if __name__ == "__main__":
    main()
