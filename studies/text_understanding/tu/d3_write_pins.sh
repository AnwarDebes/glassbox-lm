#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
# 3d: writes tu/d3/pins.sh (code and small inputs; d3_chain.sh checks it before every step) and tu/d3/pins_big.sh
# (the large inputs; checked at the start, before the fit steps and before the feature check), one
# "check <file> <sha256 first 16>" line per file. The hashes are read here with sha256sum, never typed. Earlier pins
# files are kept as pins_<UTC>.sh / pins_big_<UTC>.sh, never overwritten. Run: bash tu/d3_write_pins.sh
set -eu
emb=${REPO_ROOT}/studies/text_understanding
tu=$emb/tu
dl=${REPO_ROOT}/studies/decision_layer
raw=$dl/data/raw
bge=$emb/data_untrusted/bge-base-en-v1.5/a5beb1e3
SMALL=(
  "$tu/d3_chain.sh" "$tu/d3_write_pins.sh" "$tu/d3_wait_go.sh" "$tu/PLAN_3d.md" "$tu/d3_pairs.py" "$tu/d3_fit.py"
  "$tu/d3_mw.py" "$tu/d3_twins.py" "$tu/d3_feats.py" "$tu/d3_exists_tm.py" "$tu/d3_lr.py" "$tu/d3_verdict.py"
  "$dl/scripts/cpu_load.py"
  "$tu/a3_fit.py" "$tu/a3_teacher.py" "$tu/m1_feats.py" "$tu/m1_tm.py"
  "$emb/exists_tm.py" "$emb/glove_feats.py" "$emb/w1/wordtab_feats.py" "$emb/hardset.json"
  "$dl/src/dl/tm.py" "$dl/src/dl/tm_init.py" "$dl/src/dl/boolean.py" "$dl/src/dl/data.py" "$dl/src/dl/linear.py"
  "$dl/src/dl/explain.py"
  "$tu/a3/tables/a3B_d768.npz" "$tu/a3/fit_meta.json" "$emb/w1/tables/ppw_S1000_d500_e0.5.npz"
  "$tu/d3/pair_counts_c5.npz" "$tu/d3/pair_counts_summary.json"
  "$tu/c3/results_c3_val.jsonl" "$tu/c3/results_c3_lr.jsonl" "$tu/m1/results_m1_tm.jsonl" "$tu/m1/results_m1_lr.jsonl"
  "$bge/FILES.sha256"
  "$raw/clinc150_data_full.json" "$raw/banking77_train.parquet" "$raw/banking77_test.parquet"
  "$raw/banking77_label_names.json" "$raw/hwu64_train.parquet" "$raw/hwu64_val.parquet" "$raw/hwu64_test.parquet"
)
BIG=(
  "$tu/a3/T_train.f16.npy" "$tu/a3/T_held.f16.npy" "$tu/a3/c4_sent_train_f.txt" "$tu/a3/c4_sent_held_f.txt"
  "$tu/c3/feats/orig/clinc150__p32tok_K512.npz" "$tu/c3/feats/orig/clinc150__p32tok_K512.meta.json"
  "$tu/m1/feats/a3B/clinc150__p32tok_K512.npz" "$tu/m1/feats/a3B/banking77__p32tok_K512.npz"
  "$tu/m1/feats/a3B/hwu64__p32tok_K512.npz" "$bge/model.safetensors"
)
now=$(date -u +%Y%m%dT%H%M%SZ)
write() { local out=$1 name=$2; shift 2
  for f in "$@"; do [ -e "$f" ] || { echo "missing: $f" >&2; exit 1; }; done
  [ -e "$out" ] && mv -n "$out" "${out%.sh}_$now.sh"
  {
    echo "# $name for d3_chain.sh, written $(date -u +%Y-%m-%dT%H:%M:%SZ) by d3_write_pins.sh; every line stops the chain if its file changed"
    for f in "$@"; do echo "check $f $(sha256sum "$f" | cut -c1-16)"; done
  } > "$out"
  echo "wrote $out: $(grep -c '^check ' "$out") files, sha256 $(sha256sum "$out" | cut -c1-16)"; }
mkdir -p "$tu/d3"
write "$tu/d3/pins.sh" pins "${SMALL[@]}"
write "$tu/d3/pins_big.sh" "big pins" "${BIG[@]}"
