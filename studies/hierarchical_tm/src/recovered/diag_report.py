"""Report for the post-hoc robustness arms (R1 / R2 on slide 10: pre-registered on 2 Oct 2026, 23:35 CEST; S6-R1 on
slide 6: pre-registered on 3 Oct 2026, 01:08 CEST; see docs/registrations.md):
per seed, the final pass, train accuracy, dev pair AUC, partner-shuffle AUC and tuned accuracy (from the run's FINAL line);
the pre-registered reading (dev AUC >= 0.70 on >= 2 of 3 seeds); and the claim scorer on DEV with the best rule's
OR-groups (slide10.check2b --split dev, or slide6.score --split dev). For slide 6 also --split eval, only for E1-E5
firing (pre-registered on 3 Oct 2026, 01:08 CEST; the verdict is final and nothing is chosen from it).
Usage: python diag_report.py slide10|slide6 LABEL RUN_DIR ..."""
import json, os, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
out_dir = os.environ.pop("TMLM_RUN_DIR", ".")
shutil.copy(os.path.abspath(__file__), out_dir)
task, label, runs = sys.argv[1], sys.argv[2], sys.argv[3:]
rows = []
for r in runs:
    log = open(os.path.join(r, "log.txt")).read().splitlines()
    cfg = next(json.loads(l.split(" ", 1)[1]) for l in log if l.startswith("CONFIG "))
    fin = next(json.loads(l.split(" ", 1)[1]) for l in log if l.startswith("FINAL "))
    row = {"run": os.path.basename(r), "seed": cfg["seed"], "final_pass": fin["pass"], "train_acc": fin["train_acc"],
           "dev_auc": fin["dev_auc"], "dev_tuned_acc": fin["dev_tuned_acc"], "partner_shuffle_auc": fin.get("partner_shuffle_auc"),
           "learns_070": fin["dev_auc"] >= 0.70}
    print(f"\n== {label} {row['run']}: seed {row['seed']}, pass {row['final_pass']}, train acc {row['train_acc']}, "
          f"dev AUC {row['dev_auc']:.4f}, partner-shuffle AUC {row['partner_shuffle_auc']}")
    if task == "slide10":
        import slide10
        c = slide10.check2b(r, None, "dev")
        row["claim_dev"] = {k: c[k] for k in ("coverage_dev_288", "coverage_dev_216", "fires_dev_negated", "fires_dev_nearmiss",
                                              "fires_dev_broken", "fires_dev_other", "brackets_recovered_288")}
    else:
        import slide6
        c = slide6.score(r, "dev")
        row["claim_dev"] = {k: c[k] for k in c if k.startswith(("coverage_dev", "fires_dev_broken", "fires_dev_other", "brackets_recovered"))}
        e = slide6.score(r, "eval", show=False)
        row["e1_e5_fired"], row["e1_e5_which"] = e["e1_e5_fired"], e["e1_e5_which"]
        print(f"   E1-E5 (eval mode, for information only): the best rule covers {e['e1_e5_fired']} of 5 ({e['e1_e5_which']})")
    rows.append(row)
n = sum(r["learns_070"] for r in rows)
res = {"label": label, "n_runs": len(rows), "seeds_dev_auc_ge_070": n, "reading_learns": n >= 2}
print("\nDIAG", json.dumps(res))
print("\n| seed | final pass | train acc | dev AUC | partner-shuffle AUC | best rule: coverage (dev) | dev AUC >= 0.70 |\n|---|---|---|---|---|---|---|")
for r in rows:
    cov = r["claim_dev"].get("coverage_dev_288", r["claim_dev"].get("coverage_dev"))
    print(f"| {r['seed']} | {r['final_pass']} | {r['train_acc']} | {r['dev_auc']:.3f} | {r['partner_shuffle_auc']:.3f} | {cov:.1%} | {r['learns_070']} |")
json.dump({"runs": {"value": rows, "label": "measured"}, **{k: {"value": v, "label": "derived"} for k, v in res.items()}},
          open(os.path.join(out_dir, "metrics.json"), "w"), indent=1)
