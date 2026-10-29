"""A2: independent recompute of the reading of an a2_run.py run (the verify step of the A2 look).

    python scripts/a2_recompute.py --run results/<run folder> --out <file.json> [--rescore]

Default: reads only the files of the run folder (logp.npz, the per-token natural log
probabilities of every arm that a2_run.py saves since 2026-10-25; metrics.json), no data file.
It recomputes the perplexity of every arm, the gain and the margin of the run's code, and the
interval of bits(code) - bits(random mean) with a bootstrap of its own (numpy PCG64 seed 20261025,
10,000 resamples of the evaluation blocks; a2_run.py uses seed 0 and 1,000), and compares them
with metrics.json: every perplexity within 1e-12 relative, the same gain, margin and success and
kill readings, the sha256 of logp.npz as recorded. Interval ends may differ by the bootstrap.
--rescore: also rebuilds every arm from the saved codes in data/a2_codes and scores the run's
split again with a driver of its own (the model code of src/tally/floor and the codes are shared
with a2_run.py), and compares every per-token log probability with logp.npz (within 1e-9). For a
dev-test run this second scoring belongs to the same registered look, not to a new one; it is
refused unless the run is the finished registered look: its folder name ends in
_a2_devtest_final and its source_hashes.json holds registrations/A2.md with the sha256 of the
freeze record. Splits other than train, calib, valid and the run's split are dropped right after
loading. For a dev-test run it also checks that every file the freeze record of A2 pins (the
registration, every file under src/, the extra files) still has its recorded sha256, so that
nothing changed during the look. The result goes to --out (JSON, with the UTC time); its sha256 is
printed.
For a dev-test run it also requires the code exchange and the registered label in metrics.json, and
KN5 alone within 5e-5 bits per word of T0's 7.1760 on the same dev-test (a difference means the split
differs from T0's). Any false check sets all_match false and the exit code 1.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

STUDY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(STUDY, "src"))
from tally import ptb  # noqa: E402
from tally.floor import cache, codes, mix  # noqa: E402
from tally.floor.classlm import ClassMember  # noqa: E402
from tally.floor.kn import KN  # noqa: E402

HALF_LIVES = (25, 50, 100, 200, 400, 800, 1600, 3200, np.inf)
LINES = {"success_gain": 0.05, "kill_gain": 0.02, "margin": 0.01}
REGISTRATION = os.path.join("registrations", "A2.md")
N_BOOT, SEED = 10_000, 20261025
CLAIM_CODE = "exchange"                                   # named in registrations/A2.md
LOOK_LABEL = "registered reading of A2 on devtest"
T0_KN5_DEVTEST_BPW, KN5_TOL = 7.1760, 5e-5               # T0's Kneser-Ney 5-gram on the same dev-test


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def boot(d, blocks, n_boot=N_BOOT, seed=SEED):
    """Mean of d with a percentile interval from resampling whole blocks (token-weighted)."""
    ub, inv = np.unique(blocks, return_inverse=True)
    S, n = np.bincount(inv, weights=d), np.bincount(inv).astype(np.float64)
    idx = np.random.Generator(np.random.PCG64(seed)).integers(0, len(ub), size=(n_boot, len(ub)))
    stat = S[idx].sum(1) / n[idx].sum(1)
    return {"mean": float(d.sum() / len(d)), "lo": float(np.quantile(stat, 0.025)),
            "hi": float(np.quantile(stat, 0.975)), "n_boot": n_boot, "seed": seed, "blocks": int(len(ub))}


def reading(lp, blocks, code, masks):
    """The A2 reading of `code` from per-token log probabilities: lp[arm] arrays."""
    ppl = {k: float(np.exp(-v.mean())) for k, v in lp.items()}
    rnd = sorted(k for k in lp if k.startswith("random_s"))
    gain = 1 - ppl[code] / ppl["none"]
    margin = 1 - ppl[code] / float(np.mean([ppl[r] for r in rnd]))
    lp_r = np.mean([lp[r] for r in rnd], 0)
    iv = boot((lp_r - lp[code]) / np.log(2), blocks)
    out = {"code": code, "ppl": ppl, "gain": float(gain), "margin": float(margin), "code_minus_random_mean_bits": iv,
           "success": bool(gain >= LINES["success_gain"] and margin >= LINES["margin"] and iv["hi"] < 0),
           "kill": bool(gain < LINES["kill_gain"]), "subsets_no_verdict": {}}
    for name, m in masks.items():
        out["subsets_no_verdict"][name] = {
            "tokens": int(m.sum()),
            "code_minus_none_bits": boot((lp["none"][m] - lp[code][m]) / np.log(2), blocks[m]),
            "code_minus_random_mean_bits": boot((lp_r[m] - lp[code][m]) / np.log(2), blocks[m])}
    return out


def compare(rec, metrics):
    R, rd = metrics["arms"], metrics["reading"]
    rel = {k: abs(rec["ppl"][k] / R[k]["all"]["ppl"] - 1) for k in rec["ppl"]}
    checks = {
        "same_arms": sorted(rec["ppl"]) == sorted(R),
        "ppl_within_1e-12": bool(rel) and max(rel.values()) < 1e-12,
        "same_code": rd["best_code"] == rec["code"],
        "gain_equal": abs(rd["gain_vs_none"] - rec["gain"]) < 1e-12,
        "margin_equal": abs(rd["margin_vs_random_mean"] - rec["margin"]) < 1e-12,
        "same_success": rd["success_line"] == rec["success"],
        "same_kill": rd["kill_line"] == rec["kill"]}
    return {"max_rel_ppl_diff": max(rel.values()) if rel else None, "checks": checks,
            "run_interval": [rd["best_minus_random_mean_bits"]["lo"], rd["best_minus_random_mean_bits"]["hi"]],
            "all_match": all(checks.values())}


def rescore_allowed(run_dir, split, study=STUDY):
    """Empty list when a second scoring of the run's split is allowed."""
    if split != "devtest":
        return []
    errs = []
    if not os.path.basename(os.path.normpath(run_dir)).endswith("_a2_devtest_final"):
        errs.append("a dev-test rescore needs the registered look's folder (*_a2_devtest_final)")
    try:
        with open(os.path.join(run_dir, "run.json")) as f:
            state = json.load(f).get("state")
        with open(os.path.join(run_dir, "source_hashes.json")) as f:
            src = json.load(f)
    except OSError:
        return errs + ["the run folder has no run.json or source_hashes.json"]
    if state != "finished":
        errs.append(f"the look is not finished (state {state})")
    frozen = os.path.join(study, "registrations", "A2.frozen.json")
    if not os.path.exists(frozen):
        return errs + ["registrations/A2.frozen.json missing"]
    with open(frozen) as f:
        h = json.load(f)["registration"].get(REGISTRATION)
    if not h or src.get(REGISTRATION) != h:
        errs.append("the run does not record registrations/A2.md with the hash of its freeze record")
    return errs


def record_matches(study=STUDY):
    """Files that differ from the freeze record of A2 now (empty list: all equal); None without a record.
    Run after the look, it shows that nothing the record pins changed during the look."""
    frozen = os.path.join(study, "registrations", "A2.frozen.json")
    if not os.path.exists(frozen):
        return None
    with open(frozen) as f:
        rec = json.load(f)
    errs = []
    pinned = dict(rec.get("registration", {}), **rec.get("extra", {}))
    for p, h in pinned.items():
        f = os.path.join(study, p)
        if not os.path.exists(f) or sha256_file(f) != h:
            errs.append(p)
    now = {}
    for d, dirs, files in os.walk(os.path.join(study, "src")):
        dirs[:] = sorted(x for x in dirs if x != "__pycache__")
        for fn in files:
            if fn.endswith((".py", ".sh", ".md", ".json", ".toml")):
                q = os.path.join(d, fn)
                now[os.path.relpath(q, study)] = sha256_file(q)
    errs += [p for p in sorted(set(now) | set(rec.get("src", {}))) if now.get(p) != rec.get("src", {}).get(p)]
    return errs


def rescore(cfg, split):
    """Every arm rebuilt from the saved codes and scored on `split`: {arm: per-token log probabilities}."""
    vocab, w2i, ids, blocks = ptb.load()
    for d in (ids, blocks):
        for k in [k for k in d if k not in {"train", "calib", "valid", split}]:
            del d[k]
    V, eos = len(vocab), w2i["<eos>"]
    tr = ids["train"]
    C = {}
    names = ["exchange", "kmeans200", "predefined", "topic", "orgroups"] + [f"random_s{s}" for s in range(1, 11)]
    for name in names:
        with open(os.path.join(codes.CODE_DIR, name + ".json")) as f:
            meta = json.load(f)
        c = np.load(os.path.join(codes.CODE_DIR, name + ".npy"))
        assert codes.sha256_bytes(np.ascontiguousarray(c, dtype=np.int32).tobytes()) == meta["code_sha256"], name
        assert meta["source"]["train_stream_sha256"] == codes.stream_hash(tr), name
        C[name] = c.astype(np.int64)
    ex = [("exchange", l) for l in range(C["exchange"].shape[1])]
    tp = [("topic", l) for l in range(C["topic"].shape[1])]
    arms = {"none": [], "exchange": ex, "kmeans200": [("kmeans200", 0)],
            "predefined": [("predefined", l) for l in range(C["predefined"].shape[1])], "topic": tp,
            "kind_topic": ex + tp, "orgroups": [("orgroups", 0)]}
    for l in range(C["exchange"].shape[1]):
        arms[f"exchange_level{l}"] = [("exchange", l)]
    for s in range(1, 11):
        arms[f"random_s{s}"] = [(f"random_s{s}", l) for l in range(C[f"random_s{s}"].shape[1])]
    kn5 = KN(tr, V, eos, order=5, mode="sent")
    pk = {s: np.exp(kn5.logprob(ids[s])) for s in sorted({"calib", "valid", split})}
    best = None
    for hl in HALF_LIVES:
        for skip in ((), (eos,)):
            c = cache.Cache(V, hl, "block", skip)
            lam, _ = mix.em(np.stack([pk["calib"], c.prob(ids["calib"], blocks["calib"], pk["calib"])[0]], 1))
            ll = float(np.log(np.stack([pk["valid"], c.prob(ids["valid"], blocks["valid"], pk["valid"])[0]], 1)
                              @ lam).sum())
            if best is None or ll > best[0]:
                best = (ll, c)
    cch = best[1]
    if cch.config() != cfg["cache_chosen"]:
        raise SystemExit(f"rescore: cache choice {cch.config()} differs from the run's {cfg['cache_chosen']}")
    pc = {s: cch.prob(ids[s], blocks[s], pk[s])[0] for s in ("calib", split)}
    P = {}
    for m in sorted({m for v in arms.values() for m in v}):
        cm = ClassMember(tr, C[m[0]][:, m[1]], eos, 5, 0.5)
        P[m] = {s: np.exp(cm.logprob(ids[s])) for s in ("calib", split)}
    out = {}
    for arm, mem in arms.items():
        lam, _ = mix.em(np.stack([pk["calib"], pc["calib"]] + [P[m]["calib"] for m in mem], 1))
        out[arm] = np.log(np.stack([pk[split], pc[split]] + [P[m][split] for m in mem], 1) @ lam)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run folder of scripts/a2_run.py")
    ap.add_argument("--out", required=True, help="where the JSON result is written")
    ap.add_argument("--rescore", action="store_true")
    a = ap.parse_args()
    with open(os.path.join(a.run, "config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(a.run, "metrics.json")) as f:
        metrics = json.load(f)
    split = cfg["split"]
    if a.rescore:
        errs = rescore_allowed(a.run, split, STUDY)
        if errs:
            sys.exit("refused:\n  " + "\n  ".join(errs))
    npz = os.path.join(a.run, "logp.npz")
    z = np.load(npz)
    lp = {k[len("arm__"):]: z[k] for k in z.files if k.startswith("arm__")}
    blocks = z["blocks"]
    masks = {"unseen_bigram": z["unseen_bigram"], "duplicates_removed": ~z["duplicate"],
             "train_seen_word": z["train_seen_word"]}
    code = CLAIM_CODE if split == "devtest" else metrics["reading"]["best_code"]
    rec = reading(lp, blocks, code, masks)
    out = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "run": os.path.normpath(a.run), "split": split,
           "logp_npz_sha256": sha256_file(npz), "logp_npz_sha256_recorded": metrics.get("logp_npz_sha256"),
           "recompute": rec, "comparison": compare(rec, metrics)}
    out["comparison"]["checks"]["logp_npz_as_recorded"] = out["logp_npz_sha256"] == out["logp_npz_sha256_recorded"]
    kn5 = float(-z["kn5_alone"].mean() / np.log(2))
    out["kn5_alone_bits"] = kn5
    out["comparison"]["checks"]["kn5_alone_as_recorded"] = abs(kn5 - metrics["kn5_alone"]["bits_per_token"]) < 1e-12
    if split == "devtest":                       # the registered look: its code, its label, T0's split
        out["comparison"]["checks"]["code_is_exchange"] = metrics["reading"]["best_code"] == CLAIM_CODE
        out["comparison"]["checks"]["registered_label"] = metrics["reading"].get("label") == LOOK_LABEL
        out["comparison"]["checks"]["kn5_alone_equals_T0"] = abs(kn5 - T0_KN5_DEVTEST_BPW) < KN5_TOL
    if split == "devtest":                       # after the look: nothing the freeze record pins has changed
        diff = record_matches(STUDY)
        out["freeze_record_differences"] = diff
        out["comparison"]["checks"]["freeze_record_still_matches"] = diff == []
    if a.rescore:
        cfg2 = {"cache_chosen": metrics["cache"]["chosen"]}
        rs = rescore(cfg2, split)
        diff = {k: float(np.abs(rs[k] - lp[k]).max()) for k in lp}
        out["rescore"] = {"max_abs_logp_diff": max(diff.values()), "per_arm": diff,
                          "within_1e-9": all(v <= 1e-9 for v in diff.values())}
        out["comparison"]["checks"]["rescore_within_1e-9"] = out["rescore"]["within_1e-9"]
    out["comparison"]["all_match"] = all(out["comparison"]["checks"].values())
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print(json.dumps({"out": a.out, "sha256": sha256_file(a.out), "all_match": out["comparison"]["all_match"],
                      "checks": out["comparison"]["checks"]}))
    sys.exit(0 if out["comparison"]["all_match"] else 1)


if __name__ == "__main__":
    main()
