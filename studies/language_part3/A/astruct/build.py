"""Cross-fitted rows (review 2) and their storage, shared by S2a and the A/A rebuild (S2c).

Folds: whole fit dialogues, 5 folds, tally floor/aa.py folds() with seed 1 (the two A/A rebuilds of K8: seeds 2 and
3, each after dropping 5% of the fit dialogues with aa.subsample_train of the same seed). For the fit
positions of fold f, every count-derived
number (the floor's candidates and probabilities, top-1, entropy, the counted rates) comes from counts on the fit
replies of the other four folds; the floor's EM weights (F2 / F6) are refitted on tune-cal for each such floor.
Tune and final rows use the floor and the rates counted on all of fit (the S1 recipe).
Leak self-check (review 2; amendment 1 adds the floor part): for every training position whose act-slot set S occurs
once in fit, the act-set rate equals its back-off value, every S-keyed context lookup of the fold's floor (every level
of an F1 chain, the first level of F5) is unseen, and S's count in the fold's counting replies (the F1 / F6 bucket) is
0.
"""

from __future__ import annotations

import numpy as np

from .data import Corpus
from .floors import FloorSet, _lookup_count, chain_inputs, summarize
from .paths import add_study_paths
from .positions import Positions
from .rates import Rates
from .rows import build_rows
from .store import POS_KEYS, concat_rows, load_rows, save_rows, subset_rows  # noqa: F401

add_study_paths(tally=True)
from tally.floor import aa

N_FOLDS = 5
FOLD_SEED, AA_SEED, AA_DROP = 1, 2, 0.05


def em_names(floor: str, picks: dict) -> list:
    if floor.startswith("F2n"):
        return [floor]
    if floor == "F5":
        return [picks["F2"]]
    if floor == "F6":
        return [picks["F2"], "F6"]
    return []


def make_floor(c: Corpus, count_replies: np.ndarray, s1: dict, P_cal: Positions, log=None) -> FloorSet:
    fs = FloorSet(c, count_replies, picks=s1["picks"], need=(s1["floor"],), log=log)
    vc = fs.view(P_cal)
    for n in em_names(s1["floor"], s1["picks"]):
        fs.fit_em(vc, n)
    return fs


def folds_of(c: Corpus, fit_replies: np.ndarray, seed: int) -> np.ndarray:
    d = c.a["r_dlg"][fit_replies]
    ud, inv = np.unique(d, return_inverse=True)
    return aa.folds(ud, N_FOLDS, seed)[inv.reshape(-1)]


def aa_fit(c: Corpus, fit_replies: np.ndarray, seed: int = AA_SEED) -> tuple[np.ndarray, np.ndarray]:
    """The A/A fit set: fit replies of the dialogues kept by aa.subsample_train (5% of the dialogues dropped)."""
    d = c.a["r_dlg"][fit_replies]
    ud = np.unique(d)
    _, _, dropped = aa.subsample_train(ud, ud, seed, AA_DROP)
    keep = ~np.isin(d, dropped)
    return fit_replies[keep], dropped


def attach(rows: dict, P: Positions, summ: dict) -> dict:
    rows.update({"p_true": summ["p_true"], "top1": summ["top1"], "ent": summ["ent"], "pos_reply": P.reply,
                 "pos_t": P.t, "pos_dlg": P.dlg, "pos_x": P.x})
    return rows


def rows_for(c: Corpus, fs: FloorSet, rates: Rates | None, replies: np.ndarray, floor: str, bits: bool,
             keep_full: np.ndarray | None = None) -> tuple[dict, dict]:
    P = Positions(c, replies)
    s = summarize(fs, fs.view(P), floor, keep_full=keep_full)
    return attach(build_rows(c, P, s, rates, bits=bits), P, s), s


def s_keyed_levels(name: str, n_levels: int) -> list:
    """Levels of a floor chain whose context holds the act-slot set S: every level of F1 (the condition stays at every
    level) and the first level of F5, (R, S, w-2, w-1). F2 chains are keyed on A, F3 on R (and A)."""
    if name.startswith("F1n"):
        return list(range(n_levels))
    if name == "F5":
        return [0]
    return []


def s_lookups(fs: FloorSet, P: Positions, mask: np.ndarray) -> dict:
    """For the positions in mask: how many S-keyed context lookups of the floor set there are and how many are unseen
    (-1), and at how many positions S's count in the counting replies is 0."""
    idx = np.flatnonzero(mask)
    out = {"positions": int(len(idx)), "count_zero": 0, "lookups": 0, "unseen": 0, "chains": []}
    if not len(idx):
        return out
    out["count_zero"] = int((_lookup_count(fs.S_cnt, P.S[idx]) == 0).sum())
    for name, levels in sorted(fs.chains.items()):
        lv = s_keyed_levels(name, len(levels))
        if not lv:
            continue
        Ts, ex, _ = chain_inputs(name, P)
        sub = np.stack([levels[j].lookup(Ts[j][idx], ex[j][idx]) for j in lv])
        out["lookups"] += int(sub.size)
        out["unseen"] += int((sub == -1).sum())
        out["chains"].append(name)
    return out


def crossfit(c: Corpus, fit: np.ndarray, fold: np.ndarray, s1: dict, P_cal: Positions, train_reply: np.ndarray,
             bits: bool, log=None, leak_test: bool = True) -> dict:
    """Twin rows for every fit position, bit rows for the training replies, all cross-fitted by fold."""
    floor = s1["floor"]
    all_S = c.a["S_hash"][fit]
    uS, nS = np.unique(all_S, return_counts=True)
    single_S = uS[nS == 1]
    twin, train = [], []
    leak = {"rows": 0, "equal": 0, "true_rows": 0, "true_rows_leak_without_crossfit": 0, "S_positions": 0,
            "S_positions_count_zero": 0, "S_lookups": 0, "S_lookups_unseen": 0, "S_chains": []}
    full_rates = Rates(c, fit) if (bits and leak_test) else None
    for f in range(N_FOLDS):
        cnt, tgt = fit[fold != f], fit[fold == f]
        fs = make_floor(c, cnt, s1, P_cal)
        rt, _ = rows_for(c, fs, None, tgt, floor, bits=False)
        twin.append(rt)
        if bits:
            rates = Rates(c, cnt)
            tr = tgt[train_reply[fold == f]]
            rb, _ = rows_for(c, fs, rates, tr, floor, bits=True)
            train.append(rb)
            if leak_test:
                # review 2 (i): a training row whose act-slot set occurs once in fit gets the back-off value
                m = np.isin(c.a["S_hash"][rb["pos_reply"][rb["pos"]]], single_S)
                S = c.a["S_hash"][rb["pos_reply"][rb["pos"][m]]]
                A = c.a["A_hash"][rb["pos_reply"][rb["pos"][m]]]
                bo = rates.backoff_of_set(A, rb["cand"][m])
                leak["rows"] += int(m.sum())
                leak["equal"] += int((rb["rate_set"][m] == bo).sum())
                # the same rows with rates counted on all of fit (no cross-fitting): the true word's rate is 1
                yt = rb["y"][m]
                leak["true_rows"] += int(yt.sum())
                r_full = full_rates.rate_set(S[yt], A[yt], rb["cand"][m][yt])
                leak["true_rows_leak_without_crossfit"] += int((r_full == 1.0).sum())
                # amendment 1: the fold's floor has never seen these act-slot sets (S-keyed contexts and S counts)
                Ptr = Positions(c, tr)
                sl = s_lookups(fs, Ptr, np.isin(Ptr.S, single_S))
                leak["S_positions"] += sl["positions"]
                leak["S_positions_count_zero"] += sl["count_zero"]
                leak["S_lookups"] += sl["lookups"]
                leak["S_lookups_unseen"] += sl["unseen"]
                leak["S_chains"] = sorted(set(leak["S_chains"]) | set(sl["chains"]))
        if log:
            log(f"fold {f}: counting replies {len(cnt)}, scored {len(tgt)}, twin rows {len(rt['y'])}"
                + (f", training rows {len(train[-1]['y'])}" if bits else ""))
    return {"twin": concat_rows(twin), "train": concat_rows(train) if bits else None, "leak": leak}
