"""R4 LEARNING-CREDIT CONTROLS AND DIAGNOSTICS (docs/12 F29 items 2 and 3; docs/13 A7 H17.10, A8 items 1 and 6).

REPORTED rows with NO threshold, next to the registered H17.10 statistic of experiments/ssl_r4_images.py (which stays
the only H17.10 verdict). Config: experiments/cvtm/configs/ssl_r4_controls.json (confirmatory seeds 1 to 3) and
ssl_r4_controls_smoke.json (build smoke). Every R4 parameter (store, engine, F1, epochs, young fraction, probe
budgets, draws, LR grid, folds) is read from the R4 config the control config names (`r4_config`), never copied.

This file only IMPORTS the running R4 driver (experiments/ssl_r4_images.py as R4), the R1 control study
(experiments/ssl_r1_controls.py as C1, for the twin thresholds, the literal-freezing view and the census) and
tmlm.ssl / tmlm.engine; none of them is edited. The registered probe path is reused call for call: R4.label_draws,
R4.load_codes, SparseDocs.to_csr("log1p", l2=True), R4.take and tmlm.ssl.transfer.lr_cell with the R4 config's
lr_max_iter, c_grid and folds and seed = draw, so every control cell pairs with the registered cell of the same
(seed, budget, draw).

Stages (each its own logged run; --gpu-run is a registered `ssl_r4_images.py --stage gpu` run of one seed):
  replay   (GPU) retrain arm "main" with the registered settings and seed; epoch-1 ladder checkpoints after
           ceil(young_fraction x nb) batches (the registered C0y, "c0y"), round(5 percent) ("pct5"), round(10
           percent) ("pct10") of the nb epoch-1 batches and at the end of epoch 1 ("epoch1"). REPLAY CHECK: the
           engine config, the targets, the C0y state hash, every epoch's state hash, alive count, dev masked top-1
           and DIAG counters, and the final state hash must equal the gpu run's logged values (a mismatch is a failed
           check: exit 1); the C0y codes must equal the registered codes_main_young.npz array for array. Per batch
           the TA slot transitions are counted on the device (A8 item 6, see slot_transitions). Image codes of every
           rung in the registered format (codes_ladder_<rung>.npz: dev images plus every labelled pool image of the
           seed's draws, pool document frequency).
  twin     (GPU) the same arm with p_ins = 0, p_neg = 0, s = 2^62 (every literal-changing path of docs/11 section 5
           off; seeding, recycling, votes and vetoes on; the path list with code references is in
           experiments/cvtm/configs/ssl_r1_controls.json "literal_paths"), E epochs, the literal-freezing checks of
           the R1 twins at every epoch end and per batch, final-bank codes codes_twin_final.npz.
  randbank (GPU or CPU, --engine) the A8 random conjunction bank: C clauses, 1 to 3 positive exact (position-bound)
           literals each, drawn as tmlm.ssl.confirm.random_bank draws them (same stream) but with the position drawn
           uniformly from the pool images' 49 patch positions (the pretraining target positions), so literals come
           with the image tokens' literal frequencies; no training; codes_randbank.npz (fired clause x quadrant).
  probe    (CPU) the registered LR probe on every rung, the twin and the random bank, same draws, cells.jsonl.
  diag     (GPU) A8 item 6 per bank (registered main.final, main.young, randkm.final; the ladder; the twin; the random
           bank): included literals per clause, negated and bag shares, owned clauses per class against T, TA
           transitions crossing the include boundary per epoch (replayed arms only), class-sum histograms at the dev
           masked positions (the engine's vote sums, the closest analogue of a TM class sum).
  report   (CPU) code_main minus every rung, minus max(ladder), minus the twin, minus the random bank at 10 and 100
           labels per class: two-level bootstrap (seeds, then draws within seed; 10,000 resamples), seed-level
           Student-t interval, the driver's flat paired bootstrap; the label "literal-learning credit" (A8 item 1)
           iff code_main beats max(ladder) AND the twin by >= 1.0 point with the two-level interval excluding 0 at
           the H17.10 budget (10 per class); masked-patch top-1 per bank; the registered H17.10 statistic next to it.

Launch commands: see the config's "launch" block (one lane per seed, CPU slot first, GPU lock inside).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
for _q in (ROOT / "src", EXPERIMENTS):
    if str(_q) not in sys.path:
        sys.path.insert(0, str(_q))

import ssl_r1_controls as C1  # noqa: E402  (imported, never edited)
import ssl_r4_images as R4  # noqa: E402  (imported, never edited)
from tmlm.common import hashing, runlog  # noqa: E402
from tmlm.engine.config import INCLUDE_THRESHOLD, LIT_EMPTY, EngineConfig  # noqa: E402
from tmlm.ssl import features as F  # noqa: E402
from tmlm.ssl import patch_tokens as PT  # noqa: E402

T0 = time.time()
DEFAULT_CONFIG = ROOT / "experiments" / "cvtm" / "configs" / "ssl_r4_controls.json"
STAGES = ("replay", "twin", "randbank", "probe", "diag", "report")
RUNGS = ("c0y", "pct5", "pct10", "epoch1")
SOURCE = "ssl_r4_controls"


# ------------------------------------------------------------------ bookkeeping
class Rec:
    """Labelled metrics of one run (metrics.json through runlog), checks, log lines."""

    def __init__(self, status: str) -> None:
        self.status = status
        self.metrics: dict = {}
        self.failed: list[str] = []

    def log(self, msg: str) -> None:
        print(f"[{time.time() - T0:8.1f} s] {msg}", flush=True)

    def m(self, name: str, value, label: str = "measured", **extra) -> None:
        if isinstance(value, (np.floating, np.integer)):
            value = value.item()
        self.metrics[name] = {"value": value, "label": label, "status": self.status, **extra}

    def check(self, name: str, ok: bool, **extra) -> None:
        self.m(f"check.{name}", bool(ok), **extra)
        if not ok:
            self.failed.append(name)
            self.log(f"CHECK FAILED: {name} {extra}")

    def flush(self) -> None:
        if runlog.current_run_dir() is not None:
            runlog.write_metrics(self.metrics, on_conflict="replace")


_SCRATCH: list[Path] = []


def out_dir(stage: str) -> Path:
    d = runlog.current_run_dir()
    if d is not None:
        return Path(d)
    if not _SCRATCH:
        s = Path(os.environ.get("SSL_SCRATCH", "/tmp")) / f"ssl_r4c_{stage}_{os.getpid()}_{int(time.time())}"
        s.mkdir(parents=True, exist_ok=True)
        _SCRATCH.append(s)
    return _SCRATCH[0]


def rp(path: str | None) -> Path | None:
    if path is None or not str(path).strip():
        return None
    q = Path(str(path).strip())
    return q if q.is_absolute() else ROOT / q


def read_metrics(run: Path) -> dict:
    """{name: value} of a run's metrics.json."""
    m = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    return {k: (v["value"] if isinstance(v, dict) and "value" in v else v) for k, v in m.items()}


def load_config(path: str | Path) -> tuple[dict, dict]:
    """(control config, the R4 config it names)."""
    cj = json.loads(Path(path).read_text(encoding="utf-8"))
    r4 = json.loads((ROOT / cj["r4_config"]).read_text(encoding="utf-8"))
    return cj, r4


def data_root(args) -> Path | None:
    return Path(args.data_root) if args.data_root else None


# ------------------------------------------------------------------ pure helpers (unit tested)
def ladder_points(nb: int, young_fraction: float, percents) -> dict[str, int]:
    """Rung name -> number of epoch-1 batches after which it is taken (the end of epoch 1 is "epoch1" = nb).
    c0y = ceil(young_fraction x nb) exactly as ssl_r4_images.py computes its young bank (max(1, ceil(f x nb)));
    pctP = round half up of nb x P / 100 in exact integers, at least 1."""
    pts = {"c0y": max(1, int(math.ceil(float(young_fraction) * int(nb))))}
    for pct in percents:
        pts[f"pct{int(pct)}"] = max(1, (int(nb) * int(pct) + 50) // 100)
    pts["epoch1"] = int(nb)
    return pts


def unique_rungs(points: dict[str, int]) -> dict[int, list[str]]:
    """batch -> rung names taken there (a rung equal to an earlier one shares its state, its checkpoint and codes)."""
    out: dict[int, list[str]] = {}
    for name, b in points.items():
        out.setdefault(int(b), []).append(name)
    return dict(sorted(out.items()))


def twin_config(cfg: EngineConfig, overrides: dict) -> EngineConfig:
    return dataclasses.replace(cfg, p_ins=float(overrides["p_ins"]), p_neg=float(overrides["p_neg"]),
                               s=float(2 ** int(overrides["s_log2"])))


def slot_transitions(xp, prev: tuple, cur: tuple) -> dict[str, int]:
    """TA slot changes between two snapshots (lit_id u32 [C, L], lit_state u8 [C, L], birth_step u32 [C]) taken at
    consecutive batch ends (A8 item 6 at batch granularity). A slot is RETAINED when it tracks the same literal of
    the same clause incarnation (same lit_id, not LIT_EMPTY, same birth_step) at both ends. Counts:
      retained_changed   retained slots whose state changed (net change over the batch)
      unit_steps_lb      sum of |state change| over retained slots (a lower bound of the +-1 TA steps)
      cross_up/down      retained slots whose state crossed INCLUDE_THRESHOLD upward / downward
      lowered            retained slots whose state went down
      out, out_included  tracked literals that left their slot (freed, evicted, clause recycled), of which included
      in_, in_included   literals that entered a slot (insertion at 123, seeding at 136), of which included
    Transitions that cancel inside one batch are invisible at this granularity."""
    pid, pst, pb = prev
    cid, cst, cb = cur
    occ_p = pid != LIT_EMPTY
    occ_c = cid != LIT_EMPTY
    same = occ_p & (pid == cid) & (pb[:, None] == cb[:, None])
    ps = pst.astype(xp.int32)
    cs = cst.astype(xp.int32)
    inc_p = ps >= INCLUDE_THRESHOLD
    inc_c = cs >= INCLUDE_THRESHOLD
    d = cs - ps
    vals = xp.stack([
        (same & (d != 0)).sum(), xp.where(same, xp.abs(d), 0).sum(),
        (same & ~inc_p & inc_c).sum(), (same & inc_p & ~inc_c).sum(), (same & (d < 0)).sum(),
        (occ_p & ~same).sum(), (occ_p & ~same & inc_p).sum(), (occ_c & ~same).sum(), (occ_c & ~same & inc_c).sum(),
    ]).astype(xp.int64)
    v = vals.get() if hasattr(vals, "get") else np.asarray(vals)
    keys = ("retained_changed", "unit_steps_lb", "cross_up", "cross_down", "lowered", "out", "out_included", "in_", "in_included")
    return {k: int(x) for k, x in zip(keys, v.tolist())}


def transition_summary(t: dict[str, int]) -> dict:
    """Per-epoch fractions of the summed slot_transitions counts."""
    cross = t.get("cross_up", 0) + t.get("cross_down", 0)
    return {**t, "cross": cross,
            "frac_retained_changes_crossing": cross / t["retained_changed"] if t.get("retained_changed") else None,
            "frac_unit_steps_crossing_ub": cross / t["unit_steps_lb"] if t.get("unit_steps_lb") else None}


def replay_compare(logged: dict, ours: dict, epochs: int) -> list[tuple[str, Any, Any]]:
    """(name, ours, logged) of every value the replay must reproduce; `logged` = read_metrics(gpu run) values,
    `ours` = {"engine_config", "targets_sha256", "store_sha256", "c0y": {"batches", "state_hash", "dev_masked_top1"},
    "epochs": {k: {"state_hash", "alive", "dev_masked_top1", "diag"}}, "final_state_hash"}."""
    pairs: list[tuple[str, Any, Any]] = [
        ("engine_config", ours["engine_config"], logged.get("engine_config")),
        ("targets_sha256", ours["targets_sha256"], logged.get("targets.sha256")),
        ("store_sha256", ours["store_sha256"], logged.get("store_sha256")),
        ("epochs_logged", int(epochs), sum(1 for k in logged if k.startswith("main.epoch") and k[len("main.epoch"):].isdigit())),
        ("c0y.state_hash", ours["c0y"]["state_hash"], logged.get("main.young.state_hash")),
        ("c0y.dev_masked_top1", ours["c0y"]["dev_masked_top1"], logged.get("main.young.dev_masked_top1")),
    ]
    for k in range(1, int(epochs) + 1):
        got, ref = ours["epochs"].get(k, {}), logged.get(f"main.epoch{k}") or {}
        for f in ("state_hash", "alive", "dev_masked_top1", "diag"):
            pairs.append((f"epoch{k}.{f}", got.get(f), ref.get(f)))
    pairs.append(("final.state_hash", ours["final_state_hash"], (logged.get("main.final.codes") or {}).get("state_hash")))
    return pairs


def replay_failures(pairs: list[tuple[str, Any, Any]]) -> list[str]:
    return [name for name, a, b in pairs if a is None or b is None or a != b]


def random_bank_at(tok: np.ndarray, doc_start: np.ndarray, kind_map: np.ndarray, positions: np.ndarray, vocab: int,
                   n_clauses: int, seed: int, max_lit: int = 3):
    """tmlm.ssl.confirm.random_bank (mode masked) with the position of every literal draw taken uniformly from
    `positions` instead of from all corpus positions; everything else (stream SeedSequence([20260925, 0xA8, 3, seed]),
    n_lit uniform on 1..max_lit, column uniform among unused offsets, redraw when no literal is true) is identical,
    so positions = arange(N) gives CF.random_bank exactly (tests/test_ssl_r4_controls.py)."""
    from tmlm.ssl import confirm as CF
    from tmlm.ssl import controls as K

    tok = np.asarray(tok)
    posv = np.asarray(positions, np.int64)
    Np = int(posv.shape[0])
    lo, hi = K.doc_bounds(np, np.asarray(doc_start))
    off, kind = CF._col_tables("masked")
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
            p = posv[rng.integers(0, Np, size=pending.size)]
            c = rng.integers(0, n_cols, size=pending.size)
            used = np.where(col[pending, :t] >= 0, off[np.maximum(col[pending, :t], 0)], 1 << 30) if t else np.zeros((pending.size, 0), np.int64)
            clash = np.any(used == off[c][:, None], axis=1) if t else np.zeros(pending.size, bool)
            s = CF.symbol_at(tok, lo, hi, kind_map, p, off[c], kind[c], vocab)
            ok = ~clash & (s >= 0)
            col[pending[ok], t] = c[ok]
            sym[pending[ok], t] = s[ok]
            redraws += int((~ok).sum())
            pending = pending[~ok]
    info = {"n_clauses": C, "max_lit": int(max_lit), "n_lit_counts": np.bincount(n_lit, minlength=int(max_lit) + 1)[1:].tolist(),
            "redraws": redraws, "n_positions": Np,
            "rule": "position uniform over the given positions, column uniform among unused offsets, redraw when no literal is true"}
    return CF.RandomBank("masked", int(vocab), col, sym, n_lit, int(seed), info)


def randbank_codes(xp, firer, tok, doc_start, n_images: int, chunk_images: int, keep: np.ndarray | None = None,
                   df_rows: np.ndarray | None = None):
    """The image code of ssl_r4_images.image_codes for a random bank: counts of (fired clause, quadrant) per image
    over its 49 patch positions, rows of the images with keep[i] in ascending order, u16 counts; df = number of
    images with df_rows[i] where each column is non-zero. Returns (SparseDocs, info, df int64 [4C])."""
    from tmlm.ssl import controls as K

    C = int(firer.bank.n)
    keep = np.ones(n_images, bool) if keep is None else np.asarray(keep, bool)
    df_rows = np.ones(n_images, bool) if df_rows is None else np.asarray(df_rows, bool)
    tok_d = xp.asarray(np.asarray(tok))
    lo, hi = K.doc_bounds(xp, np.asarray(doc_start))
    keep_d, dfr_d = xp.asarray(keep), xp.asarray(df_rows)
    df = xp.zeros(4 * C, dtype=xp.int64)
    parts, fired_total, n_pos = [], 0, 0
    t0 = time.time()
    for i0 in range(0, int(n_images), int(chunk_images)):
        i1 = min(int(n_images), i0 + int(chunk_images))
        pos = xp.asarray(PT.patch_positions(np.arange(i0, i1)).astype(np.int64))
        rows, cl = firer.fire(tok_d, lo, hi, pos)
        n_f = int(cl.shape[0])
        fired_total += n_f
        n_pos += int(pos.shape[0])
        kept = np.flatnonzero(keep[i0:i1])
        if n_f:
            p = pos[rows]
            img = p // PT.TOKENS_PER_IMAGE
            o = p % PT.TOKENS_PER_IMAGE
            q = 2 * ((o // PT.ROW_LEN) >= 4).astype(xp.int64) + ((o % PT.ROW_LEN) >= 4).astype(xp.int64)
            key = (img << 32) | (cl.astype(xp.int64) * 4 + q)
            u, cnt = xp.unique(key, return_counts=True)
            uimg = u >> 32
            ucol = u & 0xFFFFFFFF
            dsel = dfr_d[uimg]
            if bool(dsel.any()):
                df += xp.bincount(ucol[dsel], minlength=4 * C)
            ksel = keep_d[uimg]
            h = (lambda a: a.get() if hasattr(a, "get") else np.asarray(a))
            rows_img, ids, c = h(uimg[ksel]), h(ucol[ksel]).astype(np.uint32), h(cnt[ksel])
            if c.size and int(c.max()) > 0xFFFF:
                raise SystemExit("a (clause, quadrant) count exceeds u16")
            c = c.astype(np.uint16)
        else:
            rows_img, ids, c = np.zeros(0, np.int64), np.zeros(0, np.uint32), np.zeros(0, np.uint16)
        rank = np.full(i1 - i0, -1, np.int64)
        rank[kept] = np.arange(kept.shape[0])
        ip = np.zeros(kept.shape[0] + 1, np.int64)
        if rows_img.size:
            ip[1:] = np.cumsum(np.bincount(rank[rows_img - i0], minlength=kept.shape[0]))
        parts.append(F.SparseDocs(ip, ids, c, C * 4))
    wall = time.time() - t0
    dfh = df.get() if hasattr(df, "get") else np.asarray(df)
    return F.merge_rows(parts), {"wall_s": wall, "positions": n_pos, "fired_per_position": fired_total / max(1, n_pos),
                                 "positions_per_s": n_pos / max(1e-9, wall), "rows_kept": int(keep.sum())}, dfh


def randbank_reference_codes(bank, tok, doc_start, kind_map, image_ids) -> F.SparseDocs:
    """Pure-Python reference (tiny inputs): CF.random_bank_bruteforce on the patch positions, aggregated by
    patch_tokens.image_code_from_events; one row per image of image_ids."""
    from tmlm.ssl import confirm as CF

    pos = PT.patch_positions(np.asarray(image_ids)).astype(np.int64)
    bf = CF.random_bank_bruteforce(bank, tok, doc_start, kind_map, pos.tolist())
    ev_p = np.concatenate([np.full(len(x), p, np.int64) for p, x in zip(pos.tolist(), bf)] + [np.zeros(0, np.int64)])
    ev_j = np.concatenate([np.asarray(x, np.int64) for x in bf] + [np.zeros(0, np.int64)])
    n_img = int(np.asarray(doc_start).shape[0] - 1)
    row = np.full(n_img, -1, np.int64)
    row[np.asarray(image_ids, np.int64)] = np.arange(len(image_ids))
    return PT.image_code_from_events(ev_p, ev_j, row, len(image_ids), bank.n)


def owned_per_class(st: dict, V: int, T: int) -> dict:
    """Alive clauses with a positive vote for each class (docs/11 'owns'), against T."""
    alive = np.asarray(st["alive"]).astype(bool)
    vc = np.asarray(st["vote_cls"])[alive]
    vw = np.asarray(st["vote_w"])[alive]
    own = (vc != 0xFFFF) & (vw > 0)
    per = np.bincount(vc[own].astype(np.int64), minlength=int(V))[: int(V)]
    q = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    return {"T": int(T), "classes": int(V), "quantiles": dict(zip([str(x) for x in q], np.quantile(per, q).tolist())),
            "mean": float(per.mean()), "classes_owned_by_none": int((per == 0).sum()),
            "classes_below_T": int((per < int(T)).sum()), "classes_at_least_T": int((per >= int(T)).sum()),
            "median_over_T": float(np.median(per) / int(T)), "total_ownerships": int(per.sum())}


def class_sum_edges(max_log2: int = 20) -> np.ndarray:
    """Signed log2 bin edges -inf, -2^max, ..., -2, -1, 0, 1, 2, ..., 2^max, inf (numpy bins: [e_i, e_i+1),
    the last one closed)."""
    pw = 2.0 ** np.arange(0, int(max_log2) + 1)
    return np.concatenate([[-np.inf], -pw[::-1], [0.0], pw, [np.inf]])


def class_sum_summary(gold: np.ndarray, best: np.ndarray, other: np.ndarray, T: int, max_log2: int = 20) -> dict:
    """Histograms and quantiles of the gold-class vote sum, the top vote sum and the margin gold minus the best
    other class, over the evaluated positions; fractions against T (the engine clips feedback at +-T)."""
    edges = class_sum_edges(max_log2)
    q = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    out = {"n_positions": int(gold.shape[0]), "T": int(T), "bin_edges": [float(e) if np.isfinite(e) else str(e) for e in edges]}
    for name, v in (("gold", gold), ("top", best), ("margin", gold.astype(np.int64) - other.astype(np.int64))):
        v = np.asarray(v, np.float64)
        out[name] = {"hist": np.histogram(v, bins=edges)[0].tolist(),
                     "quantiles": dict(zip([str(x) for x in q], np.quantile(v, q).tolist())) if v.size else None,
                     "frac_ge_T": float((v >= T).mean()) if v.size else None,
                     "frac_le_minus_T": float((v <= -T).mean()) if v.size else None,
                     "frac_zero": float((v == 0).mean()) if v.size else None}
    return out


def census_randbank(bank) -> dict:
    n = np.asarray(bank.n_lit, np.int64)
    return {"alive": int(bank.n), "included_per_clause_mean": float(n.mean()),
            "included_per_clause_hist": np.bincount(n, minlength=13).tolist(),
            "frac_literals_negated": 0.0, "frac_clauses_with_negated": 0.0, "frac_literals_bag": 0.0,
            "frac_clauses_with_bag": 0.0, "note": "positive exact literals only by construction (A8 item 1)"}


# ------------------------------------------------------------------ report statistics (unit tested)
def cell_table(cells: list[dict]) -> dict:
    """(rep, seed, budget, draw) -> accuracy of the LR cells."""
    out = {}
    for c in cells:
        if c.get("probe") == "lr":
            out[(c["rep"], int(c["seed"]), int(c["budget"]), int(c["draw"]))] = float(c["acc"])
    return out


def diff_units(acc: dict, main: str, ref: str | dict, budget: int) -> dict[int, np.ndarray]:
    """{seed: per-draw differences main minus ref in POINTS (100 x accuracy)}, draws paired within seed. `ref` is a
    rep name or {seed: rep name} (max(ladder) is chosen per seed)."""
    seeds = sorted({s for (r, s, b, d) in acc if r == main and b == budget})
    units = {}
    for s in seeds:
        rr = ref if isinstance(ref, str) else ref.get(s)
        if rr is None:
            continue
        draws = sorted(d for (r, ss, b, d) in acc if r == main and ss == s and b == budget and (rr, s, b, d) in acc)
        if draws:
            units[s] = np.array([100.0 * (acc[(main, s, budget, d)] - acc[(rr, s, budget, d)]) for d in draws])
    return units


def max_ladder(acc: dict, rungs, budget: int) -> dict[int, str]:
    """Per seed the rung with the highest mean accuracy over its draws at this budget (ties: earlier rung)."""
    seeds = sorted({s for (r, s, b, d) in acc if r in rungs and b == budget})
    out = {}
    for s in seeds:
        best, bm = None, None
        for r in rungs:
            v = [a for (rr, ss, b, d), a in acc.items() if rr == r and ss == s and b == budget]
            if v and (bm is None or float(np.mean(v)) > bm):
                best, bm = r, float(np.mean(v))
        if best is not None:
            out[s] = best
    return out


def row_stats(units: dict[int, np.ndarray], n_boot: int, boot_seed: int) -> dict:
    from tmlm.ssl import confirm as CF

    flat = np.concatenate([units[s] for s in sorted(units)]) if units else np.zeros(0)
    return {"two_level_bootstrap": CF.two_level_bootstrap(units, n_boot=int(n_boot), seed=int(boot_seed)),
            "seed_t": CF.seed_t_interval(units) if units else None,
            "paired_bootstrap_flat": R4.paired_bootstrap(flat, n_boot=int(n_boot)) if flat.size else None,
            "per_seed_mean": {str(s): float(v.mean()) for s, v in sorted(units.items())},
            "pairs": int(flat.size)}


def credit_label(ladder: dict, twin: dict, seeds_expected: int, threshold: float) -> dict:
    """A8 item 1 on the single image code: 'literal-learning credit' iff code_main minus max(ladder) AND code_main
    minus the twin are each >= threshold points with the two-level bootstrap interval excluding 0 (lo > 0)."""
    from tmlm.ssl import confirm as CF

    comp = {"ladder": CF.verdict_min_ci("main_minus_max_ladder", ladder.get("two_level_bootstrap"), ladder.get("seed_t"),
                                        threshold, seeds_expected),
            "twin": CF.verdict_min_ci("main_minus_twin", twin.get("two_level_bootstrap"), twin.get("seed_t"),
                                      threshold, seeds_expected)}
    if any(c["verdict"] == CF.INCOMPLETE for c in comp.values()):
        return {"label": CF.INCOMPLETE, "components": {k: c["verdict"] for k, c in comp.items()}}
    ok = all(c["verdict"] == CF.MET for c in comp.values())
    return {"label": "literal-learning credit" if ok else "no literal-learning credit", "credited": bool(ok),
            "components": {k: c["verdict"] for k, c in comp.items()}, "threshold_points": float(threshold)}


# ------------------------------------------------------------------ shared data of the GPU stages
def gpu_run_identity(gpu_dir: Path, r4_config: Path, rec: Rec) -> tuple[dict, int]:
    lm = read_metrics(gpu_dir)
    if lm.get("stage") != "gpu" or "main.young.state_hash" not in lm:
        raise SystemExit(f"{gpu_dir} is not a finished ssl_r4_images.py --stage gpu run with arm main")
    seed = int(lm["seed"])
    m = json.loads((gpu_dir / "metrics.json").read_text())
    cfg_sha = m.get("config_file", {}).get("sha256")
    rec.check("gpu_run.config_equals_r4_config", cfg_sha == hashing.sha256_file(r4_config), gpu_run_config_sha256=cfg_sha)
    failed = [k for k, v in lm.items() if k.startswith("check.") and v is False]
    rec.check("gpu_run.no_failed_checks", not failed, failed=failed)
    rec.m("gpu_run", str(gpu_dir))
    rec.m("seed", seed)
    return lm, seed


def load_r4_data(r4P: dict, args, seed: int, rec: Rec) -> dict:
    """The prologue of ssl_r4_images.stage_gpu, line for line: store, targets after F1, pool tokens, dev positions,
    the frozen hidden bitmap, and the rows kept in the code files (dev plus the seed's labelled pool images)."""
    ca, extras, meta = PT.load_store(r4P["corpus"], int(r4P["n_codes"]) + 1, data_root(args), check=True)
    V = int(ca.kind_map.shape[1])
    n_img = int(ca.doc_start.shape[0] - 1)
    dev = extras["dev"].astype(bool)
    pool_img, dev_img = np.flatnonzero(~dev), np.flatnonzero(dev)
    tgt_all = np.asarray(ca.targets)
    if r4P["f1"]:
        keep = PT.f1_keep(tgt_all, ca.tok[tgt_all.astype(np.int64)], t=float(r4P["f1_t"]))
        targets = tgt_all[keep].astype(np.uint32)
    else:
        targets = tgt_all
    pool_tok = ca.tok.reshape(n_img, PT.TOKENS_PER_IMAGE)[pool_img].reshape(-1)
    Q = r4P["probe"]
    draws = R4.label_draws(extras["labels"].astype(np.int64), pool_img, seed, Q["budgets"], int(Q["draws"]))
    keep_rows = dev.copy()
    for ids in draws.values():
        keep_rows[ids] = True
    dev_pos = PT.patch_positions(dev_img)
    rec.m("store_sha256", meta["sha256"])
    rec.m("targets.sha256", hashing.sha256_array(targets))
    rec.m("targets.after_f1", int(targets.shape[0]))
    rec.m("codes.rows_kept", int(keep_rows.sum()))
    return {"ca": ca, "extras": extras, "meta": meta, "V": V, "n_img": n_img, "dev": dev, "pool_img": pool_img,
            "dev_img": dev_img, "targets": targets, "pool_tok": pool_tok, "keep_rows": keep_rows,
            "keep_ids": np.flatnonzero(keep_rows), "dev_pos": dev_pos, "hidden_eval": PT.frozen_hidden(int(ca.tok.shape[0])),
            "draws": draws}


def main_engine_config(r4P: dict, V: int, seed: int) -> EngineConfig:
    ecfg = dict(r4P["engine"])
    ecfg.update({"mode": "masked", "vocab": V, "seed": int(seed)})
    return EngineConfig(**ecfg)


def arm_corpus(D: dict, arm: str, seed: int):
    from tmlm.data.synthetic import sym_freq_of
    from tmlm.engine.api import CorpusArrays

    ca = D["ca"]
    km = np.asarray(ca.kind_map) if arm == "main" else PT.random_kind_map(np.asarray(ca.kind_map), seed)
    sf = sym_freq_of(D["pool_tok"], km, D["V"])
    return CorpusArrays(tok=np.asarray(ca.tok), doc_start=np.asarray(ca.doc_start), kind_map=km,
                        stop=np.asarray(ca.stop), sym_freq=sf, targets=D["targets"]), sf


def snapshot(eng) -> tuple:
    d = eng.d
    return (d["lit_id"].copy(), d["lit_state"].copy(), d["birth_step"].copy())


def train_arm(eng, cfg: EngineConfig, epochs: int, saves: dict[int, list[str]], out: Path, rec: Rec, D: dict,
              twin: bool) -> dict:
    """The training loop of ssl_r4_images.stage_gpu (same train_batch order, the same synchronize and state reads,
    the same per-epoch masked top-1 call) plus epoch-1 rung checkpoints at `saves`, device-side slot transition
    counts per batch, and (twin) the literal-freezing checks. Returns rung states and hashes, per-epoch records."""
    from tmlm.engine.api import state_hash_of
    from tmlm.engine.checkpoint import save_checkpoint

    prefix = "twin" if twin else "main"
    nb = eng.n_batches()
    res: dict[str, Any] = {"rungs": {}, "epochs": {}, "transitions": {}}
    prev_snap = snapshot(eng)
    prev_view = None
    h = None
    for e in range(int(epochs)):
        t0 = time.time()
        diag: dict[str, int] = {}
        tr: dict[str, int] = {}
        t_obs = 0.0
        twin_bad = 0
        for b in range(nb):
            d = eng.train_batch(e, b)
            for k, v in d.items():
                diag[k] = diag.get(k, 0) + int(v)
            names = saves.get(b + 1) if e == 0 else None
            if names:
                eng.synchronize()
                tc = time.time()
                path = out / f"ladder_{names[0]}.npz"
                hh = save_checkpoint(path, eng, 0, b + 1, compress=True,
                                     extra={"rungs": names, "batches": b + 1, "batches_in_epoch": nb})
                st = eng.state()
                for n in names:
                    res["rungs"][n] = {"state": st, "hash": hh, "batches": b + 1, "file": path.name}
                rec.m(f"{prefix}.rung.{names[0]}.state_hash", hh, batches=b + 1, aliases=names, file=path.name)
                t_obs += time.time() - tc
            tc = time.time()
            eng.synchronize()
            cur = snapshot(eng)
            s = slot_transitions(eng.cp, prev_snap, cur)
            prev_snap = cur
            for k, v in s.items():
                tr[k] = tr.get(k, 0) + v
            if twin:
                twin_bad += s["cross_down"] + s["lowered"] + s["cross_up"] + (s["in_"] - s["in_included"])
            t_obs += time.time() - tc
        eng.synchronize()
        wall = time.time() - t0 - t_obs
        st = eng.state()
        if e + 1 == int(epochs):
            h = save_checkpoint(out / f"{prefix}_final.npz", eng, e + 1, 0, compress=True)
        else:
            h = state_hash_of(st, cfg)
        top1 = R4.masked_top1(eng, D["dev_pos"], D["hidden_eval"])
        rec_e = {"wall_s": wall, "observer_s": t_obs, "examples_per_s": diag.get("n_examples", 0) / max(1e-9, wall),
                 "alive": int(st["alive"].sum()), "state_hash": h, "dev_masked_top1": top1, "diag": diag}
        res["epochs"][e + 1] = rec_e
        res["transitions"][e + 1] = transition_summary(tr)
        rec.m(f"{prefix}.epoch{e + 1}", rec_e)
        rec.m(f"{prefix}.epoch{e + 1}.transitions", res["transitions"][e + 1],
              note="slot_transitions summed over the batches of the epoch (batch granularity; docstring of slot_transitions)")
        view = C1.literal_view(st)
        stats = C1.literal_compare(prev_view, view)
        rec.m(f"{prefix}.epoch{e + 1}.literals", stats, note="vs the previous epoch end (same clause, same birth step); R1 control study definitions")
        if twin:
            rec.check(f"twin.literals_frozen.epoch{e + 1}", C1.twin_frozen(stats), **stats)
            rec.check(f"twin.per_batch_no_literal_change.epoch{e + 1}", twin_bad == 0, violations=int(twin_bad),
                      rule="per batch: no retained slot lowered or crossing 128, every entering literal included (seeded at 136)")
        prev_view = view
        rec.log(f"{prefix} epoch {e + 1}: {wall:.1f} s, alive {int(st['alive'].sum())}, dev masked top-1 {top1:.4f}, "
                f"crossings {res['transitions'][e + 1]['cross']}, hash {h[:12]}")
        rec.flush()
        from tmlm.common.locks import yield_gpu_if_requested
        if yield_gpu_if_requested():
            rec.log("yielded the GPU to an exclusive job")
    res["final_hash"] = h
    res["final_state"] = eng.state()
    res["sum_G"] = int(sum(int(x) for x in np.asarray(res["final_state"]["G"]).tolist()))
    return res


def write_bank_codes(eng, st: dict, name: str, state_hash: str, D: dict, r4P: dict, out: Path, rec: Rec,
                     rs: np.random.Generator) -> tuple[str, F.SparseDocs]:
    """codes_<name>.npz in the registered format (R4.image_codes, R4.codes_check, R4.save_codes)."""
    eng.load_state(st)
    codes, info, df_pool = R4.image_codes(eng, D["n_img"], int(r4P["chunk_images"]), keep=D["keep_rows"], df_rows=~D["dev"])
    ok_agg, ok_bf = R4.codes_check(eng, codes, D["keep_ids"], rs)
    rec.check(f"{name}.codes_equal_cpu_reference", ok_agg)
    rec.check(f"{name}.fired_equal_bruteforce", ok_bf)
    fname = f"codes_{name}.npz"
    sha = R4.save_codes(out / fname, codes, image_ids=D["keep_ids"], df_pool=df_pool, state_hash=state_hash)
    nnz = np.diff(codes.indptr)
    rec.m(f"{name}.codes", {**info, "file": fname, "sha256": sha, "nnz": int(codes.ids.shape[0]),
                            "distinct_per_image_mean": float(nnz.mean()), "state_hash": state_hash,
                            "columns_used": int(np.unique(codes.ids).shape[0])})
    rec.log(f"{name}: codes {info['positions_per_s']:.0f} pos/s, fired/pos {info['fired_per_position']:.1f}, distinct/image {nnz.mean():.0f}")
    return fname, codes


def codes_equal(a: Path, b: Path) -> dict:
    x, y = np.load(a), np.load(b)
    return {k: bool(np.array_equal(x[k], y[k])) for k in ("indptr", "ids", "cnt", "n_cols", "image_ids", "df_pool", "state_hash")}


# ------------------------------------------------------------------ stages
def stage_replay(args, cj: dict, r4: dict, rec: Rec) -> None:
    from tmlm.engine.cuda.cuda_engine import CudaEngine

    r4P, p = r4["params"], cj["params"]
    out = out_dir("replay")
    gpu_dir = rp(args.gpu_run)
    lm, seed = gpu_run_identity(gpu_dir, ROOT / cj["r4_config"], rec)
    E = int(args.epochs if args.epochs is not None else r4P["epochs"])
    rec.m("epochs", E)
    D = load_r4_data(r4P, args, seed, rec)
    cfg = main_engine_config(r4P, D["V"], seed)
    rec.m("engine_config", cfg.to_dict())
    corpus, sf = arm_corpus(D, "main", seed)
    rec.check("main.sym_freq_equals_store", np.array_equal(sf, np.asarray(D["ca"].sym_freq)))
    eng = CudaEngine(cfg, corpus)
    nb = eng.n_batches()
    pts = ladder_points(nb, float(r4P["young_fraction"]), p["young_ladder"]["percent_of_epoch1"])
    saves = unique_rungs(pts)
    rec.m("main.batches_per_epoch", nb)
    rec.m("ladder.points", pts, note=p["young_ladder"]["rounding"])
    rec.flush()
    res = train_arm(eng, cfg, E, saves, out, rec, D, twin=False)
    # ---- rung codes and masked top-1
    rs = np.random.default_rng(12345)
    rungs_meta = {}
    done: dict[str, str] = {}
    for b, names in saves.items():
        r = res["rungs"][names[0]]
        eng.load_state(r["state"])
        top1 = R4.masked_top1(eng, D["dev_pos"], D["hidden_eval"])
        fname, _ = write_bank_codes(eng, r["state"], f"ladder_{names[0]}", r["hash"], D, r4P, out, rec, rs)
        for n in names:
            rungs_meta[n] = {"batches": int(b), "examples": int(min(int(D["targets"].shape[0]), b * cfg.batch)),
                             "state_hash": r["hash"], "codes_file": fname, "checkpoint": r["file"],
                             "alive": int(np.asarray(r["state"]["alive"]).sum()), "dev_masked_top1": top1, "shares_state_with": names}
            done[n] = fname
        rec.flush()
    rec.m("ladder.rungs", rungs_meta)
    # ---- the replay check
    ours = {"engine_config": cfg.to_dict(), "targets_sha256": hashing.sha256_array(D["targets"]), "store_sha256": D["meta"]["sha256"],
            "c0y": {"batches": rungs_meta["c0y"]["batches"], "state_hash": rungs_meta["c0y"]["state_hash"],
                    "dev_masked_top1": rungs_meta["c0y"]["dev_masked_top1"]},
            "epochs": {k: {f: v[f] for f in ("state_hash", "alive", "dev_masked_top1", "diag")} for k, v in res["epochs"].items()},
            "final_state_hash": res["final_hash"]}
    pairs = replay_compare(lm, ours, E)
    bad = replay_failures(pairs)
    for name, a, b in pairs:
        rec.check(f"replay.{name}_equals_gpu_run", name not in bad, ours=a if name not in ("engine_config",) else "see engine_config",
                  gpu_run=b if name not in ("engine_config",) else "see the gpu run")
    young_b = json.loads((gpu_dir / "metrics.json").read_text())["main.young.state_hash"].get("batches")
    rec.check("replay.c0y_batches_equal_gpu_run", young_b == rungs_meta["c0y"]["batches"], ours=rungs_meta["c0y"]["batches"], gpu_run=young_b)
    eq = codes_equal(out / rungs_meta["c0y"]["codes_file"], gpu_dir / "codes_main_young.npz")
    rec.check("replay.c0y_codes_equal_registered_codes_main_young", all(eq.values()), **eq)
    if E == 1:
        eq = codes_equal(out / rungs_meta["epoch1"]["codes_file"], gpu_dir / "codes_main_final.npz")
        rec.check("replay.epoch1_codes_equal_registered_codes_main_final", all(eq.values()), **eq)
    rec.m("replay.verdict", "PASS" if not bad else "FAIL", label="derived", mismatches=bad)
    rec.log(f"replay check: {'PASS' if not bad else 'FAIL ' + str(bad)}")
    rec.m("memory.device_pool_bytes", int(eng.device_memory_used()), unit="bytes")
    rec.flush()


def stage_twin(args, cj: dict, r4: dict, rec: Rec) -> None:
    from tmlm.engine.cuda.cuda_engine import CudaEngine

    r4P, p = r4["params"], cj["params"]
    out = out_dir("twin")
    gpu_dir = rp(args.gpu_run)
    lm, seed = gpu_run_identity(gpu_dir, ROOT / cj["r4_config"], rec)
    E = int(args.epochs if args.epochs is not None else r4P["epochs"])
    rec.m("epochs", E)
    D = load_r4_data(r4P, args, seed, rec)
    base = main_engine_config(r4P, D["V"], seed)
    rec.check("twin.base_config_equals_gpu_run", base.to_dict() == lm.get("engine_config"))
    cfg = twin_config(base, p["twin_overrides"])
    rec.m("engine_config", cfg.to_dict())
    rec.m("twin.overrides", p["twin_overrides"])
    rec.m("engine_thresholds", C1.twin_thresholds(cfg), note="derive() and the rational s of settle: what the engine receives")
    rec.check("twin.literal_thresholds_zero", C1.twin_thresholds_ok(cfg, base), **C1.twin_thresholds(cfg))
    corpus, _ = arm_corpus(D, "main", seed)
    eng = CudaEngine(cfg, corpus)
    rec.m("twin.batches_per_epoch", eng.n_batches())
    rec.flush()
    res = train_arm(eng, cfg, E, {}, out, rec, D, twin=True)
    tot: dict[str, int] = {}
    for ep in res["epochs"].values():
        for k, v in ep["diag"].items():
            tot[k] = tot.get(k, 0) + int(v)
    rec.m("twin.totals", tot)
    rec.check("twin.no_literal_inserted", tot.get("n_inserted", 0) == 0 and tot.get("n_neg_inserted", 0) == 0,
              n_inserted=int(tot.get("n_inserted", 0)), n_neg_inserted=int(tot.get("n_neg_inserted", 0)))
    s_num = C1.twin_thresholds(cfg)["s_num"]
    rec.check("twin.decay_provably_zero", res["sum_G"] < s_num, sum_G=str(res["sum_G"]), s_num=str(s_num),
              rule="owed <= M(j) <= sum(G) < s_num implies n_q16 = floor(owed / s) = 0 in every settle of the run")
    rs = np.random.default_rng(12345)
    fname, _ = write_bank_codes(eng, res["final_state"], "twin_final", res["final_hash"], D, r4P, out, rec, rs)
    rec.m("twin.final", {"state_hash": res["final_hash"], "codes_file": fname, "checkpoint": "twin_final.npz",
                         "dev_masked_top1": res["epochs"][E]["dev_masked_top1"], "alive": res["epochs"][E]["alive"]})
    rec.m("memory.device_pool_bytes", int(eng.device_memory_used()), unit="bytes")
    rec.flush()


def stage_randbank(args, cj: dict, r4: dict, rec: Rec) -> None:
    from tmlm.ssl import confirm as CF

    r4P, p = r4["params"], cj["params"]
    rb = p["random_bank"]
    out = out_dir("randbank")
    gpu_dir = rp(args.gpu_run)
    lm, seed = gpu_run_identity(gpu_dir, ROOT / cj["r4_config"], rec)
    D = load_r4_data(r4P, args, seed, rec)
    ca = D["ca"]
    tok, ds, km = np.asarray(ca.tok), np.asarray(ca.doc_start), np.asarray(ca.kind_map)
    C = int(lm["engine_config"]["n_clauses"])
    xp = np
    if args.engine == "cuda":
        import cupy as xp  # noqa: F811
    rec.m("randbank.engine", args.engine)
    t0 = time.time()
    pos = PT.patch_positions(D["pool_img"]).astype(np.int64)
    bank = random_bank_at(tok, ds, km, pos, D["V"], C, seed, int(rb["max_literals"]))
    bank.save(out / "random_bank.npz")
    rec.m("randbank.info", bank.info)
    rec.m("randbank.sha256", hashing.sha256_file(out / "random_bank.npz"))
    rec.m("randbank.draw_positions", "the 49 patch positions of every pool image (the pretraining target positions before F1)",
          n=int(pos.shape[0]))
    single = CF.literal_single_counts(xp, tok, ds, km, "masked", D["V"])
    firer = CF.RandomBankFirer(xp, bank, single, km)
    codes, info, df_pool = randbank_codes(xp, firer, tok, ds, D["n_img"], int(rb["chunk_images"]), keep=D["keep_rows"], df_rows=~D["dev"])
    sha = R4.save_codes(out / "codes_randbank.npz", codes, image_ids=D["keep_ids"], df_pool=df_pool,
                        state_hash=hashing.sha256_file(out / "random_bank.npz"))
    nnz = np.diff(codes.indptr)
    rec.m("randbank.codes", {**info, "file": "codes_randbank.npz", "sha256": sha, "nnz": int(codes.ids.shape[0]),
                             "distinct_per_image_mean": float(nnz.mean()), "columns_used": int(np.unique(codes.ids).shape[0])})
    rec.log(f"randbank codes: {info['positions_per_s']:.0f} pos/s, fired/pos {info['fired_per_position']:.1f}, distinct/image {nnz.mean():.0f}")
    # ---- checks: a few kept images against the pure-Python reference on a clause subset; numpy vs the engine used
    rs = np.random.default_rng(778)
    sub = np.sort(rs.choice(bank.n, size=min(int(rb["check_clauses"]), bank.n), replace=False))
    small = CF.RandomBank(bank.mode, bank.vocab, bank.col[sub], bank.sym[sub], bank.n_lit[sub], bank.seed, {})
    rows = np.sort(rs.choice(D["keep_ids"].shape[0], size=min(int(rb["check_images"]), D["keep_ids"].shape[0]), replace=False))
    imgs = D["keep_ids"][rows]
    ref = randbank_reference_codes(small, tok, ds, km, imgs)
    colmap = {int(j) * 4 + qq: k * 4 + qq for k, j in enumerate(sub.tolist()) for qq in range(4)}
    ok = True
    for i, r in enumerate(rows.tolist()):
        got = codes.rows(np.asarray([r]))
        g = {colmap[int(c)]: int(n) for c, n in zip(got.ids.tolist(), got.cnt.tolist()) if int(c) in colmap}
        e = ref.rows(np.asarray([i]))
        ok &= g == {int(c): int(n) for c, n in zip(e.ids.tolist(), e.cnt.tolist())}
    rec.check("randbank.codes_equal_bruteforce", ok, images=imgs.tolist(), clauses=int(sub.shape[0]))
    if xp is not np:
        firer_np = CF.RandomBankFirer(np, bank, single, km)
        n0 = min(int(rb["chunk_images"]), D["n_img"])
        a, _, _ = randbank_codes(np, firer_np, tok, ds, n0, n0)
        b, _, _ = randbank_codes(xp, firer, tok, ds, n0, n0)
        rec.check("randbank.cuda_codes_equal_numpy_first_chunk", bool(np.array_equal(a.indptr, b.indptr) and np.array_equal(a.ids, b.ids)
                                                                      and np.array_equal(a.cnt, b.cnt)), images=n0)
    lit_ok = bool(np.all(single[np.maximum(bank.col, 0), np.maximum(bank.sym, 0)][bank.col >= 0] > 0))
    rec.check("randbank.every_literal_true_somewhere", lit_ok)
    rec.m("randbank.wall_s", time.time() - t0)
    rec.flush()


def control_reps(replay: Path | None, twin: Path | None, randbank: Path | None) -> dict[str, tuple[Path, str]]:
    """rep name -> (codes file, state hash or bank hash) of every control representation."""
    reps: dict[str, tuple[Path, str]] = {}
    if replay is not None:
        rm = read_metrics(replay)
        for n in RUNGS:
            r = rm["ladder.rungs"].get(n)
            if r:
                reps[f"ladder.{n}"] = (replay / r["codes_file"], r["state_hash"])
    if twin is not None:
        t = read_metrics(twin)["twin.final"]
        reps["twin"] = (twin / t["codes_file"], t["state_hash"])
    if randbank is not None:
        reps["randbank"] = (randbank / "codes_randbank.npz", read_metrics(randbank)["randbank.sha256"])
    return reps


def stage_probe(args, cj: dict, r4: dict, rec: Rec) -> None:
    from tmlm.ssl import transfer as TR

    r4P = r4["params"]
    Q = r4P["probe"]
    out = out_dir("probe")
    gpu_dir = rp(args.gpu_run)
    lm, seed = gpu_run_identity(gpu_dir, ROOT / cj["r4_config"], rec)
    runs = {"replay": rp(args.replay_run), "twin": rp(args.twin_run), "randbank": rp(args.randbank_run)}
    for k, d in runs.items():
        if d is None:
            continue
        m = read_metrics(d)
        rec.check(f"{k}_run.same_seed", int(m["seed"]) == seed, run=str(d))
        rec.check(f"{k}_run.same_gpu_run", rp(m["gpu_run"]) == gpu_dir, run=str(d))
        failed = [x for x, v in m.items() if x.startswith("check.") and v is False]
        if failed:
            raise SystemExit(f"{d} has failed checks {failed}; not probing it")
    reps = control_reps(runs["replay"], runs["twin"], runs["randbank"])
    if args.reps:
        want = [s.strip() for s in args.reps.split(",") if s.strip()]
        reps = {k: v for k, v in reps.items() if k in want}
    rec.m("probe.reps", {k: str(v[0]) for k, v in reps.items()})
    ca, extras, meta = PT.load_store(r4P["corpus"], int(r4P["n_codes"]) + 1, data_root(args), check=True)
    labels = extras["labels"].astype(np.int64)
    dev = extras["dev"].astype(bool)
    pool_img, dev_img = np.flatnonzero(~dev), np.flatnonzero(dev)
    y_dev = labels[dev_img]
    n_img = int(ca.doc_start.shape[0] - 1)
    # one representation at a time (loaded, probed over every budget and draw, then freed): the eager load of every
    # representation held all CSR matrices at once and passed 32 GiB; each cell's inputs and LR call are unchanged
    def load_rep(rep: str) -> tuple:
        f, h = reps[rep]
        sha = hashing.sha256_file(f)
        rec.m(f"rep.{rep}.file_sha256", sha, state_hash=h, file=str(f))
        sd, image_ids, df_pool = R4.load_codes(f)
        row_of = np.full(n_img, -1, np.int64)
        row_of[image_ids] = np.arange(image_ids.shape[0])
        return (sd.to_csr("log1p", l2=True), False, row_of, df_pool)  # exactly ssl_r4_images.representations
    draws = R4.label_draws(labels, pool_img, seed, Q["budgets"], int(Q["draws"]))
    cache: dict = {}
    n_cells = 0
    cells = out / "cells.jsonl"
    with open(cells, "a", encoding="utf-8") as fh:
        for rep in reps:
            R = load_rep(rep)
            rec.flush()
            for budget in Q["budgets"]:
                for draw in range(int(Q["draws"])):
                    lab = draws[(int(budget), draw)]
                    yl = labels[lab]
                    base = {"seed": seed, "budget": int(budget), "draw": draw, "labelled_sha256": hashing.sha256_bytes(lab.astype("<i8").tobytes())}
                    key = (reps[rep][1], int(budget), draw)
                    if key in cache:
                        r, reused = cache[key], True
                    else:
                        r = TR.lr_cell(R4.take(R, lab), yl, R4.take(R, dev_img), y_dev, seed=draw, dense=R[1],
                                       max_iter=int(Q["lr_max_iter"]), c_grid=tuple(Q["c_grid"]), max_folds=int(Q["folds"]))
                        cache[key], reused = r, False
                    row = {**base, "rep": rep, "probe": "lr", "acc": r.accuracy, "choice": r.choice, "cv": r.cv, "fit_s": r.fit_s,
                           "pred_sha256": hashing.sha256_bytes(np.asarray(r.pred, np.int64).astype("<i8").tobytes()),
                           "state_hash": reps[rep][1], "source": SOURCE, "reused_identical_state": reused}
                    fh.write(json.dumps(row, sort_keys=True, default=float) + "\n")
                    fh.flush()
                    n_cells += 1
                    rec.log(f"budget {budget} draw {draw} {rep} lr: {r.accuracy:.4f} (C {r.choice}, {r.fit_s:.1f} s{', reused' if reused else ''})")
                    rec.m("probe.cells", n_cells)
            del R
            rec.flush()
    rec.m("probe.cells_sha256", hashing.sha256_file(cells))
    rec.flush()


def class_sums(eng, positions: np.ndarray, hidden: np.ndarray, tok: np.ndarray, chunk: int = 4096) -> tuple:
    """(gold, top, best-other vote sums, top-1) at the positions, the forward pass of ssl_r4_images.masked_top1."""
    cp = eng.cp
    hid = cp.asarray(hidden)
    g_all, t_all, o_all = [], [], []
    hits = 0
    for lo in range(0, positions.shape[0], chunk):
        pos = positions[lo:lo + chunk]
        eng._sync_state_struct()
        fw = eng._forward(R4.EVAL_EPOCH, cp.asarray(pos), hid, 0)
        eng._marks = []
        sc = fw.scores
        gold = cp.asarray(tok[pos.astype(np.int64)].astype(np.int64))
        gs = cp.take_along_axis(sc, gold[:, None], axis=1)[:, 0]
        top = sc.max(axis=1)
        sc2 = sc.astype(cp.int64)
        sc2[cp.arange(sc.shape[0]), gold] = -(1 << 40)
        oth = sc2.max(axis=1)
        pred = cp.argmax(sc, axis=1)
        hits += int((pred == gold).sum())
        g_all.append(gs.get())
        t_all.append(top.get())
        o_all.append(oth.get())
        del fw, sc, sc2
    return np.concatenate(g_all), np.concatenate(t_all), np.concatenate(o_all), hits / max(1, positions.shape[0])


def stage_diag(args, cj: dict, r4: dict, rec: Rec) -> None:
    from tmlm.ssl import confirm as CF
    from tmlm.engine.checkpoint import load_checkpoint
    from tmlm.engine.cuda.cuda_engine import CudaEngine

    r4P, p = r4["params"], cj["params"]
    gpu_dir = rp(args.gpu_run)
    lm, seed = gpu_run_identity(gpu_dir, ROOT / cj["r4_config"], rec)
    D = load_r4_data(r4P, args, seed, rec)
    E = sum(1 for k in lm if k.startswith("main.epoch") and k[len("main.epoch"):].isdigit())
    replay, twin, randbank = rp(args.replay_run), rp(args.twin_run), rp(args.randbank_run)
    # bank -> (checkpoint, arm, logged top-1, logged state hash, transitions per epoch or None)
    banks: dict[str, tuple] = {
        "main.final": (gpu_dir / "main_final.npz", "main", lm[f"main.epoch{E}"]["dev_masked_top1"], lm["main.final.codes"]["state_hash"], None),
        "main.young": (gpu_dir / "main_young.npz", "main", lm["main.young.dev_masked_top1"], lm["main.young.state_hash"], None),
    }
    if (gpu_dir / "randkm_final.npz").exists() and f"randkm.epoch{E}" in lm:
        banks["randkm.final"] = (gpu_dir / "randkm_final.npz", "randkm", lm[f"randkm.epoch{E}"]["dev_masked_top1"],
                                 lm["randkm.final.codes"]["state_hash"], None)
    not_measured = {"randkm.final.transitions": "the randkm arm is not replayed by this script (only arm main and its twin are)",
                    "main.young.transitions": "C0y is part of the replayed run: see ladder.c0y and replay epoch 1",
                    "randbank.owned_per_class": "no votes: a random conjunction bank owns no class",
                    "randbank.class_sums": "no votes: a random conjunction bank has no class sums and no masked top-1",
                    "randbank.transitions": "no training: no TA transitions"}
    if replay is not None:
        rm = read_metrics(replay)
        trans = {str(k): rm.get(f"main.epoch{k}.transitions") for k in range(1, E + 1)}
        banks["main.final"] = banks["main.final"][:4] + (trans,)
        for n in RUNGS:
            r = rm["ladder.rungs"].get(n)
            same = [k for k, b in banks.items() if b[3] == r["state_hash"]] if r else []
            if r and not same:
                banks[f"ladder.{n}"] = (replay / r["checkpoint"], "main", r["dev_masked_top1"], r["state_hash"],
                                        {"1": rm.get("main.epoch1.transitions")} if n == "epoch1" else None)
            elif r:
                rec.m(f"diag.ladder.{n}.same_state_as", same, note="identical state hash: diagnostics not repeated")
    if twin is not None:
        tm = read_metrics(twin)
        banks["twin.final"] = (twin / "twin_final.npz", "main", tm["twin.final"]["dev_masked_top1"], tm["twin.final"]["state_hash"],
                               {str(k): tm.get(f"twin.epoch{k}.transitions") for k in range(1, int(tm["epochs"]) + 1)})
    rec.m("diag.banks", {k: str(v[0]) for k, v in banks.items()})
    rec.m("diag.not_measured", not_measured, label="derived")
    ml = int(p["diag"]["class_sum_max_log2"])
    engines: dict[str, Any] = {}
    eng = None
    for name, (ck, arm, top1_logged, h_logged, trans) in banks.items():
        cfg, st, meta = load_checkpoint(ck)
        rec.check(f"diag.{name}.state_hash_equals_logged", meta["state_hash"] == h_logged, ours=meta["state_hash"], logged=h_logged)
        c = C1.census_of(st, int(cfg.vocab))
        c["owned_per_class"] = owned_per_class(st, int(cfg.vocab), int(cfg.T))
        key = (arm, json.dumps(cfg.to_dict(), sort_keys=True))
        if key not in engines:
            eng = None
            engines.clear()
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
            corpus, _ = arm_corpus(D, arm, seed)
            engines[key] = CudaEngine(cfg, corpus)
        eng = engines[key]
        eng.load_state(st)
        g, t, o, top1 = class_sums(eng, D["dev_pos"], D["hidden_eval"], np.asarray(D["ca"].tok))
        rec.check(f"diag.{name}.top1_equals_logged", top1 == top1_logged, ours=top1, logged=top1_logged)
        c["class_sums"] = class_sum_summary(g, t, o, int(cfg.T), ml)
        c["dev_masked_top1"] = top1
        c["transitions_per_epoch"] = trans if trans is not None else "not measured (diag.not_measured)"
        rec.m(f"diag.{name}", c)
        rec.log(f"diag {name}: alive {c['alive']}, included/clause {c['included_per_clause_mean']}, top-1 {top1:.4f}")
        rec.flush()
    if randbank is not None:
        bank = CF.RandomBank.load(randbank / "random_bank.npz")
        rec.m("diag.randbank", {**census_randbank(bank), "owned_per_class": "not defined (diag.not_measured)",
                                "class_sums": "not defined (diag.not_measured)", "transitions_per_epoch": "not defined (diag.not_measured)"})
    rec.flush()


def stage_report(args, cj: dict, r4: dict, rec: Rec) -> None:
    rp_ = cj["params"]["report"]
    budgets = [int(b) for b in rp_["budgets"]]
    seeds_expected = len(cj["params"]["seeds"])
    reg_cells, ctl_cells, top1, diag = [], [], {}, {}
    for d in [rp(x) for x in args.runs.split(",") if x.strip()]:
        m = read_metrics(d) if (d / "metrics.json").exists() else {}
        cells = [json.loads(x) for x in (d / "cells.jsonl").read_text().splitlines() if x.strip()] if (d / "cells.jsonl").exists() else []
        failed = [k for k, v in m.items() if k.startswith("check.") and v is False]
        rec.check(f"input_run_without_failed_checks.{d.name}", not failed, failed=failed)
        if m.get("controls.stage") is None:
            reg_cells += cells
            if m.get("stage") == "gpu":
                s = int(m["seed"])
                E = sum(1 for k in m if k.startswith("main.epoch") and k[len("main.epoch"):].isdigit())
                top1.setdefault(s, {})["code_main"] = m[f"main.epoch{E}"]["dev_masked_top1"]
                top1[s]["code_young"] = m["main.young.dev_masked_top1"]
        else:
            ctl_cells += [c for c in cells if c.get("source") == SOURCE]
            s = int(m["seed"])
            if m["controls.stage"] == "replay":
                for n, r in m["ladder.rungs"].items():
                    top1.setdefault(s, {})[f"ladder.{n}"] = r["dev_masked_top1"]
            elif m["controls.stage"] == "twin":
                top1.setdefault(s, {})["twin"] = m["twin.final"]["dev_masked_top1"]
            elif m["controls.stage"] == "randbank":
                top1.setdefault(s, {})["randbank"] = "not defined: no votes"
            elif m["controls.stage"] == "diag":
                diag[s] = {k[len("diag."):]: v for k, v in m.items() if k.startswith("diag.") and isinstance(v, dict) and "alive" in v}
    acc = cell_table(reg_cells)
    acc.update(cell_table(ctl_cells))
    rec.m("report.seeds", sorted({s for (_, s, _, _) in acc}), label="derived")
    # pairing checks: identical labelled samples; the C0y rung reproduces the registered code_young cells exactly
    reg_lab = {(c["seed"], c["budget"], c["draw"]): c["labelled_sha256"] for c in reg_cells if c.get("probe") == "lr"}
    bad_lab = [(c["seed"], c["budget"], c["draw"], c["rep"]) for c in ctl_cells
               if reg_lab.get((c["seed"], c["budget"], c["draw"])) != c["labelled_sha256"]]
    rec.check("labelled_samples_equal_registered", not bad_lab, mismatches=bad_lab[:20])
    reg_young = {(c["seed"], c["budget"], c["draw"]): c["pred_sha256"] for c in reg_cells if c.get("probe") == "lr" and c["rep"] == "code_young"}
    c0y = [(c["seed"], c["budget"], c["draw"], c["pred_sha256"]) for c in ctl_cells if c["rep"] == "ladder.c0y"]
    bad_c0y = [k[:3] for k in c0y if reg_young.get(k[:3]) != k[3]]
    rec.check("c0y_predictions_equal_registered_code_young", bool(c0y) and not bad_c0y, cells=len(c0y), mismatches=bad_c0y[:20])
    rungs = [f"ladder.{n}" for n in RUNGS]
    rows, labels = {}, {}
    for b in budgets:
        mx = max_ladder(acc, rungs, b)
        refs: dict[str, Any] = {f"minus_{r}": r for r in rungs}
        refs.update({"minus_max_ladder": mx, "minus_twin": "twin", "minus_randbank": "randbank",
                     "minus_code_young_registered": "code_young", "minus_bovw_registered": "bovw"})
        for name, ref in refs.items():
            units = diff_units(acc, "code_main", ref, b)
            rows[f"b{b}.{name}"] = {**row_stats(units, int(rp_["n_boot"]), int(rp_["boot_seed"])),
                                    "reference": {str(k): v for k, v in ref.items()} if isinstance(ref, dict) else ref, "budget": b, "unit": "points (100 x accuracy), code_main minus reference"}
            rec.m(f"report.b{b}.code_main_{name}", rows[f"b{b}.{name}"], label="derived")
        labels[b] = credit_label(rows[f"b{b}.minus_max_ladder"], rows[f"b{b}.minus_twin"], seeds_expected, float(rp_["label_threshold_points"]))
        rec.m(f"report.b{b}.literal_learning_label", labels[b], label="derived",
              primary=bool(b == int(rp_["label_budget"])), rule=rp_["label_rule"])
    rec.m("report.literal_learning_label", labels.get(int(rp_["label_budget"])), label="derived", rule=rp_["label_rule"])
    means = {}
    for (r, s, b, d), a in acc.items():
        means.setdefault(f"{r}.b{b}", []).append(a)
    rec.m("report.mean_accuracy", {k: {"mean": float(np.mean(v)), "n": len(v)} for k, v in sorted(means.items())}, label="derived")
    rec.m("report.registered_h17_10", {f"b{b}": R4.h17_10(reg_cells, b, margin=float(r4["params"]["verdict"]["margin"])) for b in budgets},
          label="derived", note="the registered statistic of ssl_r4_images.py --stage report, recomputed on the same cells; the only H17.10 verdict")
    rec.m("report.masked_top1_per_bank", {str(s): v for s, v in sorted(top1.items())}, label="measured",
          note="dev masked patch top-1 (ssl_r4_images.masked_top1) of every bank, from the gpu, replay and twin runs")
    rec.m("report.diag", {str(s): v for s, v in sorted(diag.items())}, label="measured")
    (out_dir("report") / "report.json").write_text(json.dumps({"rows": rows, "labels": {str(k): v for k, v in labels.items()}},
                                                              indent=1, default=float) + "\n")
    rec.log("label at the H17.10 budget: " + json.dumps(labels.get(int(rp_["label_budget"])), default=float))
    rec.flush()


# ------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--gpu-run", default=None, help="a registered ssl_r4_images.py --stage gpu run (one seed)")
    ap.add_argument("--replay-run", default=None)
    ap.add_argument("--twin-run", default=None)
    ap.add_argument("--randbank-run", default=None)
    ap.add_argument("--reps", default=None, help="probe: only these reps (comma separated), default all")
    ap.add_argument("--runs", default=None, help="report: registered gpu and probe runs plus every control run")
    ap.add_argument("--engine", default="cuda", choices=["cuda", "cpu"], help="randbank: cupy or numpy")
    ap.add_argument("--epochs", type=int, default=None, help="replay and twin: override the R4 config's epochs (smoke)")
    ap.add_argument("--data-root", default=None, help="override the data root (smoke: a scratch directory)")
    args = ap.parse_args(argv)
    cj, r4 = load_config(args.config)
    p = cj["params"]
    rec = Rec("smoke" if p.get("smoke") else "reported_control_no_threshold")
    rec.m("controls.stage", args.stage)
    rec.m("config_file", str(args.config), sha256=hashing.sha256_file(args.config))
    rec.m("r4_config", cj["r4_config"], sha256=hashing.sha256_file(ROOT / cj["r4_config"]))
    if args.epochs is not None and not p.get("smoke"):
        raise SystemExit("--epochs is a smoke override; the confirmatory controls use the R4 config's epochs")
    if args.stage in ("replay", "twin", "randbank", "probe", "diag") and not args.gpu_run:
        raise SystemExit(f"--stage {args.stage} needs --gpu-run")
    if args.stage == "report" and not args.runs:
        raise SystemExit("--stage report needs --runs")
    try:
        {"replay": stage_replay, "twin": stage_twin, "randbank": stage_randbank, "probe": stage_probe,
         "diag": stage_diag, "report": stage_report}[args.stage](args, cj, r4, rec)
    finally:
        rec.m("wall_s", time.time() - T0)
        rec.m("failed_checks", rec.failed)
        rec.flush()
    rec.log(f"stage {args.stage} done; failed checks: {rec.failed}")
    return 1 if rec.failed else 0


if __name__ == "__main__":
    sys.exit(main())
