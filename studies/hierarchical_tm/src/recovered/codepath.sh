#!/bin/bash
# final_test_report_codepath_test_exploratory: DEV data only, never --final-test.
# Tests the one branch the final test uses that the dev check could not: a shuffled-label run WITH per-pair scores.
# The "shuffled" run here is a FAKE: a copy of ns_all seed 1's dev score dir whose source CONFIG says
# shuffle_labels true and seed 101. Its AUC and CI must therefore equal seed 1's exactly. Not a result.
set -e
cp "$0" "$TMLM_RUN_DIR/"
PY=.venv-htm/bin/python
STUDY="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
R=$STUDY/results
F=$TMLM_RUN_DIR/fake_runs
mkdir -p $F/20261002T000000Z_FAKE_shuf_src $F/20261002T000001Z_FAKE_shuf_score
$PY - "$R" "$F" <<'PYEOF'
import json, shutil, sys
R, F = sys.argv[1], sys.argv[2]
src, sco = R + "/20261002T074540Z_e4s_nsall_s100_s1", R + "/20261002T082456Z_e4s_nsall_s100_s1_score"
cfg = next(json.loads(l.split(" ", 1)[1]) for l in open(src + "/log.txt") if l.startswith("CONFIG "))
cfg.update(shuffle_labels=True, seed=101)
open(F + "/20261002T000000Z_FAKE_shuf_src/log.txt", "w").write("CONFIG " + json.dumps(cfg) + "\n")
sc = next(json.loads(l.split(" ", 1)[1]) for l in open(sco + "/log.txt") if l.startswith("SCORE "))
sc["run"] = "20261002T000000Z_FAKE_shuf_src"
open(F + "/20261002T000001Z_FAKE_shuf_score/log.txt", "w").write("SCORE " + json.dumps(sc) + "\n")
shutil.copy(sco + "/scores_dev.npz", F + "/20261002T000001Z_FAKE_shuf_score/")
print("FAKE shuffled run built from", sco, "(seed 101, shuffle_labels true in its fake CONFIG)")
PYEOF
$PY -u $STUDY/src/final_test_report.py --split dev \
  --group ns_all ns_all $R/20261002T082456Z_e4s_nsall_s100_s1_score $R/20261002T084106Z_e4s_nsall_s100_s2_score $R/20261002T090952Z_e4s_nsall_s100_s3_score \
  --shuffled ns_all $F/20261002T000001Z_FAKE_shuf_score $R/20261002T075700Z_e4s_nsall_s100_shuf_s1
$PY - "$TMLM_RUN_DIR/report.json" <<'PYEOF'
import json, sys
g = json.load(open(sys.argv[1]))["groups"]["ns_all"]
s1 = next(s for s in g["seeds"] if s["seed"] == 1)
fk = next(s for s in g["shuffled"] if s["seed"] == 101)
ok = fk["auc"] == s1["auc"] and fk["auc_ci95"] == s1["auc_ci95"] and fk["equals_score_line"] is True
print("CODEPATH_TEST", "pass" if ok else "FAIL", {"fake": [fk["auc"], fk["auc_ci95"]], "seed1": [s1["auc"], s1["auc_ci95"]]})
sys.exit(0 if ok else 1)
PYEOF
