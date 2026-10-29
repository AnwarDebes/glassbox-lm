import json
for l in open("results.jsonl"):
    r = json.loads(l)
    print(r["tag"].ljust(18), r["dataset"].ljust(9), r["encoder"].ljust(11), r["view"].ljust(18), "cl", r["cfg"]["n_clauses"],
          "bits", r["n_bits"], "TM", r["val_acc"], "LRbits", r["bits_logreg_val"], "LRraw", r["raw_logreg_val"],
          "oos", r.get("oos_auroc_val_max_sum"), "train_s", r["train_s"], "best_ep", r["best_epoch"])
