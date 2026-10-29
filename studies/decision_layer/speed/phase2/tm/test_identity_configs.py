"""Bitwise identity of the three EXACT candidate configurations (configs.py: T1T3, R1, T1T3_R1) of FastCoTM
(fast_cotm.py) against the decision_layer dl.tm.CoTM, on the CPU.

Why this file: the pinned tests (glassbox-lm/studies/decision_layer/speed/tmcode/test_identity*.py, unedited) never build these exact settings.
Their variant "all" was the prototype's defaults (T1-T3 plus evalonce, pace 0), and no variant has pace > 0.
Derived from tmcode/test_identity.py: same data, shapes, seed, epochs and comparisons, original against new:
  - automata ta and weights w after training (np.array_equal),
  - the generator state after training (torch.equal on gen.get_state(): the same random numbers were drawn),
  - every dev val_acc (and its epoch) and best_epoch,
  - the integer class sums of the final machine on the dev rows, by the original code (CoTM.class_sums) and by the
    class's own class_sums,
plus: the switches in effect equal the named configuration, and no CUDA event was recorded (pace is a CPU no-op).
Shapes: unchunked (CHUNK_ELEMS 50,000,000), 2 chunks ((N // 2) * m), 5 chunks (ceil(N / 5) * m); s 16 and s 8;
N 256 clauses, T 256, batch 128, seed 1, 3 epochs; the first --rows train rows and the first 1,024 dev rows of the decision_layer
cached ABCD view-B bits, through tmcode/prof_tm.py's loader (pinned by SHA-256, read-only).
Static checks of configs.py: every configuration has exactly FastCoTM's switch keywords; a bare FastCoTM has every
switch off; R1 = every switch off plus pace > 0; T1T3 has pace 0; T1T3_R1 = T1T3 with R1's pace; evalonce and
probs_expand are off everywhere.
PLANTED-BROKEN variants, run in the same process on the first selected shape and s, must FAIL (else this test fails):
  - prob:  T1T3 with the binomial probabilities x (1 + 2^-20) (the change review/tests/broken_tm.py plants);
  - count: T1T3 whose T1 count tensor cnt_dec loses nf (the not-fired Type I events) on the plain-literal half;
  - draw:  R1 plus one extra torch.rand draw per step from the TM's generator (a "pacing" that touches the stream).

Run (CUDA hidden, 2 threads, nice 19, behind the cpu_load gate; writes only --out, which must not exist):
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 RAYON_NUM_THREADS=2 HF_HUB_OFFLINE=1 \
    PYTHONDONTWRITEBYTECODE=1 nice -n 19 <decision_layer>/.venv-dl/bin/python -I -B test_identity_configs.py \
    --group unchunked --out OUT.json [--configs T1T3,R1,T1T3_R1] [--s 16,8] [--planted prob,count,draw | none]
    [--rows 1024] [--epochs 3]
Exit 0 only if every configuration case passes, every planted case fails and the static checks hold (JSON all_pass);
exit 1 otherwise. A CPU pass is necessary, not sufficient: torch.binomial maps elements to draws differently on the
GPU, where the BG benchmarks decide.
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../../.."))

import argparse
import hashlib
import importlib.util
import inspect
import json
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TMCODE = Path(f"{_REPO_ROOT}/studies/decision_layer/speed/tmcode")
PINS = {
    HERE / "fast_cotm.py": "a72eb8b849c927464287d92b1d445ceca97c9e0b5aee24aa8d1a71e7e3a307b8",
    HERE / "configs.py": "a7205c918060f3812bd29052e2128551dda0597058a6b49ca266da13977ee5a0",
    TMCODE / "prof_tm.py": "71f1a483b16cf8bba1c93d9591d9e8463a51df6d6c950203ec006f9423f32a41",
}
NAMES = ("T1T3", "R1", "T1T3_R1")
PLANTS = ("prob", "count", "draw")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(modname: str, p: Path):
    got = sha(p)
    if got != PINS[p]:
        raise SystemExit(f"refused: {p} sha256 {got} != pinned {PINS[p]}")
    spec = importlib.util.spec_from_file_location(modname, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


prof_tm = load("speed_p2tm_prof_tm", TMCODE / "prof_tm.py")  # the decision_layer src on sys.path; torch.cuda.synchronize no-op
fc = load("speed_p2tm_fast_cotm", HERE / "fast_cotm.py")
cf = load("speed_p2tm_configs", HERE / "configs.py")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import dl.tm as T  # noqa: E402
from dl.tm import CoTM, TMConfig  # noqa: E402

FastCoTM = fc.FastCoTM
SWITCHES = tuple(n for n, p in inspect.signature(FastCoTM.__init__).parameters.items() if p.kind is p.KEYWORD_ONLY)


class PlantedProb(FastCoTM):
    """MUST FAIL: binomial probabilities moved by one part in 2^20."""

    def _prob(self, like, p):
        return super()._prob(like, p * (1.0 + 2.0 ** -20))


class PlantedCount(FastCoTM):
    """MUST FAIL: the T1 count cnt_dec without nf on the plain half (an integer slip in the rewritten counts)."""

    def _counts(self, Xd, L, g1, g1nf, g2):
        cnt_inc, cnt_dec, cnt_t2 = super()._counts(Xd, L, g1, g1nf, g2)
        if self.opt["xonly"] and self.cfg.negation:
            cnt_dec[:, : self.f] -= g1nf.sum(0, dtype=torch.float32).unsqueeze(1)
        return cnt_inc, cnt_dec, cnt_t2


class PlantedDraw(FastCoTM):
    """MUST FAIL: one extra draw per training step from the TM's own generator."""

    def _step(self, Xb, yb):
        super()._step(Xb, yb)
        torch.rand(1, generator=self.gen, device=self.dev)


PLANTED = {"prob": (PlantedProb, "T1T3"), "count": (PlantedCount, "T1T3"), "draw": (PlantedDraw, "R1")}


def static_checks() -> dict:
    sig = inspect.signature(FastCoTM.__init__).parameters
    defaults = {n: sig[n].default for n in SWITCHES}
    c = {n: dict(getattr(cf, n)) for n in NAMES}
    r1_pace = c["R1"]["pace"]
    checks = {
        "switch_keywords": list(SWITCHES),
        "keys_equal_switches": all(set(v) == set(SWITCHES) for v in c.values()),
        "bare_FastCoTM_all_off": all(v in (False, 0) for v in defaults.values()) and defaults.get("pace") == 0,
        "R1_switches_off_pace_positive": all(c["R1"][k] is False for k in SWITCHES if k != "pace") and r1_pace > 0,
        "T1T3_pace_0": c["T1T3"]["pace"] == 0,
        "T1T3_on": all(c["T1T3"][k] is True for k in ("xonly", "probs", "elem")),
        "T1T3_R1_is_T1T3_plus_R1_pace": c["T1T3_R1"] == {**c["T1T3"], "pace": r1_pace},
        "evalonce_probs_expand_off": all(v["evalonce"] is False and v["probs_expand"] is False for v in c.values()),
        "defaults": defaults,
        "configs": c,
    }
    checks["pass"] = all(v for k, v in checks.items() if k not in ("switch_keywords", "defaults", "configs"))
    return checks


def run(cls, C, f, cfg, data, chunk, opts=None):
    T.CHUNK_ELEMS = chunk
    tm = cls(C, f, cfg, device="cpu", **opts) if opts is not None else cls(C, f, cfg, device="cpu")
    Xtr, ytr, Xdv, ydv = data
    t0 = time.perf_counter()
    tm.fit(Xtr, ytr, Xdv, ydv)
    dt = time.perf_counter() - t0
    Xv = torch.as_tensor(Xdv, dtype=torch.uint8)
    S = CoTM.class_sums(tm, Xv).numpy()  # the original evaluation code
    S2 = tm.class_sums(Xv).numpy()  # the class's own (the original's too while evalonce is off)
    return tm, dt, S, S2


def same(a, b) -> dict:
    return {
        "ta": bool(np.array_equal(a.ta.numpy(), b.ta.numpy())),
        "w": bool(np.array_equal(a.w.numpy(), b.w.numpy())),
        "gen_state": bool(torch.equal(a.gen.get_state(), b.gen.get_state())),
        "history": [h["val_acc"] for h in a.history] == [h["val_acc"] for h in b.history],
        "history_epochs": [h["epoch"] for h in a.history] == [h["epoch"] for h in b.history],
        "best_epoch": getattr(a, "best_epoch", None) == getattr(b, "best_epoch", None),
    }


def compare(ref, S_ref, fast, S_o, S_f, want_opts) -> dict:
    r = same(ref, fast)
    r["class_sums_orig_code"] = bool(np.array_equal(S_ref, S_o))
    r["class_sums_fast_code"] = bool(np.array_equal(S_ref, S_f))
    r["opts_equal_config"] = fast.opt == want_opts
    r["no_cuda_event_on_cpu"] = len(fast._events) == 0
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--group", default="all", choices=["all", "unchunked", "chunks2", "chunks5"])
    ap.add_argument("--configs", default=",".join(NAMES))
    ap.add_argument("--s", default="16,8")
    ap.add_argument("--planted", default=",".join(PLANTS), help="comma list of prob,count,draw, or none")
    ap.add_argument("--rows", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"refused: {args.out} exists (use a fresh output)")
    names = [n for n in args.configs.split(",") if n]
    plants = [] if args.planted == "none" else [p for p in args.planted.split(",") if p]
    s_list = [float(x) for x in args.s.split(",") if x]
    if not names or not set(names) <= set(NAMES) or not set(plants) <= set(PLANTS) or not s_list:
        raise SystemExit(f"--configs must be a subset of {NAMES}, --planted of {PLANTS} (or none), --s non-empty")
    utc0 = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    torch.set_num_threads(2)
    st = static_checks()
    print(f"static checks {'PASS' if st['pass'] else 'FAIL'}: {st}", flush=True)
    Xtr, ytr, Xdv, ydv, C = prof_tm.load("abcd", args.rows, 1024)
    f = Xtr.shape[1]
    data = (Xtr, ytr, Xdv, ydv)
    m = 2 * f
    N = 256
    groups = [("unchunked", 50_000_000), ("chunks2", (N // 2) * m), ("chunks5", -(-N // 5) * m)]
    cases, planted = [], []
    for chunk_name, chunk in groups:
        if args.group not in ("all", chunk_name):
            continue
        for s in s_list:
            cfg = TMConfig(n_clauses=N, T=N, s=s, epochs=args.epochs, batch_size=128, seed=1)
            ref, t_ref, S_ref, _ = run(CoTM, C, f, cfg, data, chunk)
            for name in names:
                want = dict(getattr(cf, name))
                fast, t_fast, S_o, S_f = run(FastCoTM, C, f, cfg, data, chunk, want)
                r = compare(ref, S_ref, fast, S_o, S_f, want)
                ok = all(r.values())
                n_chunks = 1 if N * m <= chunk else -(-N // max(1, chunk // m))
                cases.append({"case": f"{chunk_name}/s{s:g}/{name}", "pass": ok, **r, "n_chunks": n_chunks,
                              "t_ref_s": round(t_ref, 3), "t_fast_s": round(t_fast, 3),
                              "history": [h["val_acc"] for h in ref.history],
                              "best_epoch": getattr(ref, "best_epoch", None)})
                print(f"{'PASS' if ok else 'FAIL'} {chunk_name}/s{s:g}/{name} {r} ref {t_ref:.2f}s fast {t_fast:.2f}s",
                      flush=True)
            if plants and not planted:  # once, on the first shape and s of this run, against the same reference
                for pname in plants:
                    pcls, base = PLANTED[pname]
                    want = dict(getattr(cf, base))
                    bad, _, S_o, S_f = run(pcls, C, f, cfg, data, chunk, want)
                    r = compare(ref, S_ref, bad, S_o, S_f, want)
                    detected = not all(r.values())
                    planted.append({"case": f"PLANTED-{pname}/{chunk_name}/s{s:g}/{base}", "must_fail": True,
                                    "detected": detected, **r})
                    print(f"{'MUST-FAIL ok (detected)' if detected else 'MUST-FAIL NOT DETECTED'} "
                          f"PLANTED-{pname}/{chunk_name}/s{s:g}/{base} {r}", flush=True)
    all_pass = bool(cases) and all(c["pass"] for c in cases) and all(p["detected"] for p in planted) \
        and len(planted) == len(plants) and st["pass"]
    res = {"utc_start": utc0, "utc_end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
           "torch_threads": torch.get_num_threads(), "test_sha256": sha(Path(__file__).resolve()),
           "pinned_sha256": {str(k): v for k, v in PINS.items()}, "rows": args.rows, "epochs": args.epochs,
           "group": args.group, "s": s_list, "configs_run": names, "planted_run": plants, "n_clauses": N,
           "n_features": f, "n_classes": C, "static_checks": st, "all_pass": all_pass, "n_cases": len(cases),
           "n_cases_pass": sum(c["pass"] for c in cases), "n_planted": len(planted),
           "n_planted_detected": sum(p["detected"] for p in planted), "cases": cases, "planted": planted}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1) + "\n")
    print(f"{'ALL PASS' if all_pass else 'FAIL'}: {res['n_cases_pass']} of {len(cases)} cases pass; "
          f"{res['n_planted_detected']} of {len(planted)} planted cases detected; static checks "
          f"{'pass' if st['pass'] else 'FAIL'} -> {args.out}", flush=True)
    raise SystemExit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
