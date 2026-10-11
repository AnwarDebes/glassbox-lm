"""Group recovery (gates.md, Check 2a) for finished artificial-test runs (results/*_e2_L*), from their include bits.
Usage: python pcp_recovery.py [run-dir ...]   (default: every e2 run). Prints one JSON line per run and a table."""
import glob, json, os, re, sys
import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
from pcp_world import World, recovery  # noqa: E402

R = os.path.join(os.path.dirname(SRC), "results")


def one(d):
    cfg = next(json.loads(l.split(" ", 1)[1]) for l in open(os.path.join(d, "log.txt")) if l.startswith("CONFIG "))
    m = re.search(r"pairs_(L\d)_s(\d+)", cfg["data"])
    rung, seed = m.group(1), int(m.group(2))
    z = np.load(os.path.join(d, "include_bits.npz"))
    inc = np.unpackbits(z["class1"], axis=-1, count=cfg["F"]).astype(bool)
    fin = next((json.loads(l.split(" ", 1)[1]) for l in open(os.path.join(d, "log.txt")) if l.startswith("FINAL ")), {})
    out = {"run": os.path.basename(d)[17:], "rung": rung, "seed": seed, "model": cfg["model"], "s": cfg["s"],
           "passes": fin.get("pass"), "dev_auc": round(fin.get("dev_auc", float("nan")), 3),
           "tuned_acc": fin.get("dev_tuned_acc"), "best_pass_by_dev_auc": fin.get("best_pass_by_dev_auc")}
    if cfg["model"] == "hier":
        out.update({k: (round(v, 3) if isinstance(v, float) else v)
                    for k, v in recovery(inc, World(rung, seed).groups, cfg["A"], cfg["K"]).items()})
    return out


def main():
    dirs = sys.argv[1:] or sorted(d for d in glob.glob(os.path.join(R, "*_e2_L*")) if os.path.exists(os.path.join(d, "include_bits.npz")))
    rows = [one(d) for d in dirs]
    for r in rows:
        print(json.dumps(r))
    print("\n| run | AUC | tuned acc | best pass | recovery | recovery ignoring empty | always-true share |\n|---|---|---|---|---|---|---|")
    for r in rows:
        if r["model"] == "hier":
            print(f"| {r['run']} | {r['dev_auc']} | {r['tuned_acc']} | {r['best_pass_by_dev_auc']} | {r['recovery']} | "
                  f"{r['recovery_ignoring_empty']} | {r['always_true_group_share']} |")
    run_dir = os.environ.get("TMLM_RUN_DIR")
    if run_dir:
        json.dump(rows, open(os.path.join(run_dir, "recovery.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
