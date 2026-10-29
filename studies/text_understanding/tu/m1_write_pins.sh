#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
# M1: writes tu/m1/pins.sh, one "check <file> <sha256 first 16>" line per code and input file that m1_chain.sh uses.
# The hashes are read here with sha256sum, never typed. An earlier pins.sh is kept as pins_<UTC>.sh, never overwritten.
# Run: bash tu/m1_write_pins.sh
set -eu
emb=${REPO_ROOT}/studies/text_understanding
tu=$emb/tu
dl=${REPO_ROOT}/studies/decision_layer
raw=$dl/data/raw
pot=${HOME}/.cache/huggingface/hub/models--minishlab--potion-base-32M/snapshots/1e5a03f8eeb2c98b928fbbd846f22f816360919f
out=$tu/m1/pins.sh
FILES=(
  "$tu/m1_chain.sh" "$tu/m1_feats.py" "$tu/m1_tm.py" "$tu/m1_lr.py" "$tu/m1_verdict.py" "$tu/m1_standin.py"
  "$tu/m1_check_cache.py" "$tu/m1_overlap.py" "$tu/m1_write_pins.sh" "$tu/PLAN_M1.md"
  "$emb/exists_tm.py" "$emb/w1/wordtab_feats.py" "$emb/w1/build_corpus_c4.py" "$tu/c3_common.py" "$tu/a3_teacher.py"
  "$dl/src/dl/tm.py" "$dl/src/dl/tm_init.py" "$dl/src/dl/boolean.py" "$dl/src/dl/data.py" "$dl/src/dl/linear.py"
  "$dl/src/dl/explain.py"
  "$tu/a3/tables/a3B_d768.npz" "$emb/w1/tables/ppw_S1000_d500_e0.5.npz" "$tu/c3/kept_F.jsonl" "$tu/c3x/twin_kept.jsonl"
  "$tu/c3/feats/orig/clinc150__p32tok_K512.npz" "$tu/c3/results_c3_val.jsonl" "$tu/c3/results_c3_lr.jsonl"
  "$tu/m1/overlap_m1.json" "$emb/hardset.json"
  "$raw/clinc150_data_full.json" "$raw/banking77_train.parquet" "$raw/banking77_test.parquet"
  "$raw/banking77_label_names.json" "$raw/hwu64_train.parquet" "$raw/hwu64_val.parquet" "$raw/hwu64_test.parquet"
  "$pot/model.safetensors" "$pot/modules.json" "$pot/tokenizer.json"
)
for f in "${FILES[@]}"; do [ -e "$f" ] || { echo "missing: $f" >&2; exit 1; }; done
[ -e "$out" ] && mv -n "$out" "$tu/m1/pins_$(date -u +%Y%m%dT%H%M%SZ).sh"
{
  echo "# pins for m1_chain.sh, written $(date -u +%Y-%m-%dT%H:%M:%SZ) by m1_write_pins.sh; every line stops the chain if its file changed"
  for f in "${FILES[@]}"; do echo "check $f $(sha256sum "$f" | cut -c1-16)"; done
} > "$out"
echo "wrote $out: $(grep -c '^check ' "$out") files, sha256 $(sha256sum "$out" | cut -c1-16)"
