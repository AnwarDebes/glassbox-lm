#!/usr/bin/env python
"""R15 report: TM bandit with replay against R10's TM bandit and R14's neural bandit.

Online accuracy: one number per run (cum_online_acc in abcd_bandit.jsonl); difference of means,
bootstrap over seeds (each arm's seeds resampled independently, 10,000 resamples, seed 0).
Final test accuracy: cluster bootstrap over test conversations (scripts/report_abcd.py).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(1, str(ROOT / "scripts"))

from dl.abcd import load_examples  # noqa: E402
from report_abcd import cluster_boot  # noqa: E402

BASE = ROOT / "results" / "confirm" / "R10"


def runs(learner: str, tag: str) -> list[dict]:
    rs = [json.loads(l) for l in (BASE / "abcd_bandit.jsonl").read_text().splitlines() if l.strip()]
    return sorted([r for r in rs if r["learner"] == learner and r["tag"] == tag], key=lambda r: r["seed"])


def boot_means(a: np.ndarray, b: np.ndarray, n: int = 10000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    da = a[rng.integers(0, len(a), size=(n, len(a)))].mean(1)
    db = b[rng.integers(0, len(b), size=(n, len(b)))].mean(1)
    d = da - db
    return {"diff": float(a.mean() - b.mean()), "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5))}


def main() -> None:
    te, _ = load_examples("test")
    cid = np.array([r["convo_id"] for r in te])
    arms = {"tm_replay4": ("tm", "R15"), "tm_noreplay": ("tm", "R10"), "neural": ("neural", "R14")}
    rec, corr, online = {}, {}, {}
    for name, (learner, tag) in arms.items():
        rs = runs(learner, tag)
        online[name] = np.array([r["cum_online_acc"] for r in rs])
        C = np.stack([np.load(BASE / f"preds_{learner}_{tag}_seed{r['seed']}.npz")["pred"] == np.load(BASE / f"preds_{learner}_{tag}_seed{r['seed']}.npz")["y"] for r in rs]).astype(float)
        corr[name] = C.mean(0)
        rec[name] = {"seeds": [r["seed"] for r in rs], "test_acc": float(C.mean()), "online_mean": float(online[name].mean()),
                     "online_per_seed": [round(float(x), 5) for x in online[name]], "seconds_mean": float(np.mean([r["seconds"] for r in rs])),
                     "curve_mean": [round(float(x), 4) for x in np.mean([[c["eval_acc"] for c in r["curve"]] for r in rs], 0)]}
    ones = np.ones(len(te))
    out = {"arms": rec,
           "R15.1_online_replay_minus_noreplay": boot_means(online["tm_replay4"], online["tm_noreplay"]),
           "R15.2_final_replay_minus_neural": cluster_boot(corr["tm_replay4"] - corr["neural"], ones, cid),
           "R15.3_online_replay_minus_neural": boot_means(online["tm_replay4"], online["neural"]),
           "final_replay_minus_noreplay": cluster_boot(corr["tm_replay4"] - corr["tm_noreplay"], ones, cid)}
    r1, r2, r3 = out["R15.1_online_replay_minus_noreplay"], out["R15.2_final_replay_minus_neural"], out["R15.3_online_replay_minus_neural"]
    out["R15.1"] = bool(r1["diff"] >= 0.030 and r1["lo"] > 0)
    out["R15.2"] = bool(r2["diff"] >= -0.010)
    out["R15.3"] = bool(r3["diff"] >= -0.030)
    out["R15_met"] = bool(out["R15.1"] and out["R15.2"])
    (BASE / "report_R15.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "arms"}, indent=1))
    for k, v in rec.items():
        print(k, round(v["test_acc"], 4), "online", round(v["online_mean"], 4), "s", round(v["seconds_mean"]))


if __name__ == "__main__":
    main()
