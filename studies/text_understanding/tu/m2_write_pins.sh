#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
# M2: the pins of the registered run. tu/m1_write_pins.sh (33d289fe82d1a62c) with these changes: M2's file list (no
# paraphrase, twin, overlap or old-set test files; the MTOP zip, the frozen matched list and its source overlap file,
# the code check's cache meta, cpu_load.py, the M1 parents with the saved diff, and the venv's version files added),
# full sha256 values (M1 wrote the first 16 hex characters), and, after the review of draft 1 (2026-10-25), three modes:
#   bash tu/m2_write_pins.sh --list
#       prints the file list, one absolute path per line (the GPU dry run hashes these files when it starts);
#   bash tu/m2_write_pins.sh --block tu/m2_dry/<UTC>/dry_summary.json
#       prints the sha256sum block for the plan's "## Pins" section: every file of the list except PLAN_M2.md, plus the
#       dry run's summary (read here with sha256sum, never typed);
#   bash tu/m2_write_pins.sh tu/m2_dry/<UTC>/dry_summary.json        (at the freeze; the summary is required)
#       writes tu/m2/pins.sh, one "check <file> <full sha256>" line per file of the list (PLAN_M2.md included) and the
#       summary, but only if tu/m2_freeze_check.py passes on it (the plan's block equals the pins, the plan pins itself,
#       the dry summary passed and recorded the same code); otherwise nothing is written and the exit is 1. An earlier
#       pins.sh is kept as pins_<UTC>.sh, never overwritten.
# Every path argument may be relative to the working directory: the dry summary, the plan and the output are resolved
# with realpath first, so pins.sh and the block hold absolute paths only (re-check of draft 2, MINOR D; the chain reads
# the pins from its own working directory). Run every command from glassbox-lm/studies/text_understanding, as tu/PLAN_M2.md says.
# Test hooks (tests/test_m2_chain.sh only): M2_PINS_OUT (another output path), M2_PLAN (another plan path).
set -eu
emb=${REPO_ROOT}/studies/text_understanding
tu=$emb/tu
dl=${REPO_ROOT}/studies/decision_layer
raw=$dl/data/raw
sp=$dl/.venv-dl/lib/python3.11/site-packages
pot=${HOME}/.cache/huggingface/hub/models--minishlab--potion-base-32M/snapshots/1e5a03f8eeb2c98b928fbbd846f22f816360919f
py=$dl/.venv-dl/bin/python
out=$(realpath -m "${M2_PINS_OUT:-$tu/m2/pins.sh}")
plan=$(realpath -m "${M2_PLAN:-$tu/PLAN_M2.md}")
FILES=(
  "$tu/m2_chain.sh" "$tu/m2_feats.py" "$tu/m2_tm.py" "$tu/m2_lr.py" "$tu/m2_verdict.py" "$tu/m2_standin.py"
  "$tu/m2_check_cache.py" "$tu/m2_data.py" "$tu/m2_write_pins.sh" "$tu/m2_dry_gpu.sh" "$tu/m2_audit_run.py"
  "$tu/m2_codecheck.py" "$tu/m2_freeze_check.py" "$tu/m2_code_diff.sh" "$tu/m2_code_diff_vs_m1.txt" "$plan"
  "$tu/tests/test_m2.py" "$tu/tests/test_m2_chain.sh" "$tu/tests/test_m2_data.py"
  "$tu/m1_feats.py" "$tu/m1_tm.py" "$tu/m1_lr.py" "$tu/m1_verdict.py" "$tu/m1_chain.sh" "$tu/m1_write_pins.sh"
  "$tu/m1_check_cache.py" "$tu/m1_standin.py" "$tu/archive/m2_data_draft0_f4d2a44b.py"
  "$emb/exists_tm.py" "$emb/w1/wordtab_feats.py"
  "$dl/src/dl/tm.py" "$dl/src/dl/tm_init.py" "$dl/src/dl/boolean.py" "$dl/src/dl/data.py" "$dl/src/dl/linear.py"
  "$dl/src/dl/explain.py" "$dl/scripts/cpu_load.py"
  "$tu/a3/tables/a3B_d768.npz" "$emb/w1/tables/ppw_S1000_d500_e0.5.npz"
  "$emb/data_untrusted/mtop/s3v_qLlhqQW2ypnQQXRE/mtop.zip" "$emb/fresh/mtop_test_matched_v1.json" "$emb/fresh/mtop_overlap.json"
  "$tu/c3/feats/orig/clinc150__p32tok_K512.npz" "$tu/c3/feats/orig/clinc150__p32tok_K512.meta.json"
  "$tu/c3/results_c3_val.jsonl" "$tu/c3/results_c3_lr.jsonl" "$emb/hardset.json" "$raw/clinc150_data_full.json"
  "$pot/model.safetensors" "$pot/modules.json" "$pot/tokenizer.json"
  "$sp/torch/version.py" "$sp/numpy/version.py" "$sp/sentence_transformers/__init__.py" "$sp/tokenizers/__init__.py"
  "$sp/transformers/__init__.py" "$sp/safetensors/__init__.py" "$sp/huggingface_hub/__init__.py"
)
if [ "${1:-}" = --list ]; then printf '%s\n' "${FILES[@]}"; exit 0; fi
if [ "${1:-}" = --block ]; then
  [ $# -eq 2 ] && [ -e "$2" ] || { echo "usage: m2_write_pins.sh --block DRY_SUMMARY (an existing file)" >&2; exit 2; }
  dry=$(realpath "$2")
  for f in "${FILES[@]}"; do [ "$f" = "$plan" ] && continue; [ -e "$f" ] || { echo "missing: $f" >&2; exit 1; }; done
  for f in "${FILES[@]}" "$dry"; do [ "$f" = "$plan" ] || sha256sum "$f"; done
  exit 0
fi
[ $# -eq 1 ] && [ -e "$1" ] || { echo "usage: m2_write_pins.sh DRY_SUMMARY (the GPU dry run's dry_summary.json is required)" >&2; exit 2; }
dry=$(realpath "$1")
FILES+=("$dry")
for f in "${FILES[@]}"; do [ -e "$f" ] || { echo "missing: $f" >&2; exit 1; }; done
mkdir -p "$(dirname "$out")"
tmp=$(mktemp "$(dirname "$out")/.pins_tmp.XXXXXX")
{
  echo "# pins for m2_chain.sh, written $(date -u +%Y-%m-%dT%H:%M:%SZ) by m2_write_pins.sh; every line stops the chain if its file changed"
  for f in "${FILES[@]}"; do echo "check $f $(sha256sum "$f" | cut -c1-64)"; done
} > "$tmp"
if ! "$py" -I -B "$tu/m2_freeze_check.py" --plan "$plan" --pins "$tmp" --dry "$dry"; then
  rm -f "$tmp"; echo "m2_write_pins.sh: the freeze check refused; no pins file written" >&2; exit 1
fi
[ -e "$out" ] && mv -n "$out" "$(dirname "$out")/pins_$(date -u +%Y%m%dT%H%M%SZ).sh"
mv "$tmp" "$out"
echo "wrote $out: $(grep -c '^check ' "$out") files, sha256 $(sha256sum "$out" | cut -c1-64)"
