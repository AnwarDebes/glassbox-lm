#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"  # published copy: the repository root
# Baseline fingerprints of the fixed benchmark set, from the decision_layer stored records (read-only), before any change.
#   bash bench/make_baseline.sh OUTDIR
# TM jobs B1-B6 (bench/jobs.tsv): result row in results/explore/abcd.jsonl + the saved model file.
# MILP M1 / M2: e72e's stored checks (results/explore/verify_e72e.jsonl): M1 guarded (5 seeds x 50), M2 raw (5 x 50),
# M2s1 raw seed 1 only (the 50 checks that BC4 reruns with counterexample points).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dl=${REPO_ROOT}/studies/decision_layer
py="$dl/.venv-dl/bin/python"
out="${1:?usage: make_baseline.sh OUTDIR}"
mkdir -p "$out"
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
while IFS=$'\t' read -r id tag seed _runner _argv _ts _log; do
  [[ -z "$id" || "$id" == \#* ]] && continue
  nice -n 19 "$py" -I -B "$here/fingerprint.py" tm --rows "$dl/results/explore/abcd.jsonl" --tag "$tag" --seed "$seed" \
    --model "$dl/results/explore/abcd_models/${tag}_seed${seed}.npz" --out "$out/$id.json"
done < "$here/jobs.tsv"
nice -n 19 "$py" -I -B "$here/fingerprint.py" milp --rows "$dl/results/explore/verify_e72e.jsonl" --tag e72e --guarded yes --out "$out/M1.json"
nice -n 19 "$py" -I -B "$here/fingerprint.py" milp --rows "$dl/results/explore/verify_e72e.jsonl" --tag e72e --guarded no --out "$out/M2.json"
nice -n 19 "$py" -I -B "$here/fingerprint.py" milp --rows "$dl/results/explore/verify_e72e.jsonl" --tag e72e --guarded no --seeds 1 --out "$out/M2s1.json"
nice -n 19 "$py" -I -B - "$out" <<'EOF'
import json, sys, hashlib
from pathlib import Path
out = Path(sys.argv[1])
summ = {}
for p in sorted(out.glob("[BM]*.json")):
    d = json.loads(p.read_text())
    t = d["timing"]
    summ[p.stem] = {"result_sha256": d["result_sha256"], "file_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                    "kind": d["kind"], "key": d["key"],
                    "acc": d["result"].get("acc"), "best_epoch": d["result"].get("best_epoch"),
                    "train_s": t.get("train_s"), "seconds_total": t.get("seconds_total"), "seconds_max": t.get("seconds_max"),
                    "outcomes": d["result"].get("outcome_counts"),
                    "ta_sha256": d["result"].get("model", {}).get("ta", {}).get("sha256"),
                    "w_sha256": d["result"].get("model", {}).get("w", {}).get("sha256")}
(out / "SUMMARY.json").write_text(json.dumps(summ, indent=1, sort_keys=True) + "\n")
print(json.dumps(summ, indent=1, sort_keys=True))
EOF
