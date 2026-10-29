#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"  # published copy: the repository root
# run_gpu_job.sh ROOT JOBID IMPL [--dry-cpu]    one fixed GPU benchmark job (bench/jobs.tsv) in a private root
#
#   ROOT    a run root made by bench/make_root.sh (its script copies are re-checked against the study's files)
#   JOBID   B1 ... B6 (bench/jobs.tsv: tag, seed, runner, argv exactly as recorded in the decision_layer queue logs)
#   IMPL    "orig" = the decision_layer code as it is; or a path to a NEW patch file (phase 2; e.g. one that sets
#           dl.tm_init.CoTM to a new class) which bench/with_patch.py runs before the unedited entry script
#   --dry-cpu  path check without the GPU: CUDA hidden, no lock, no go marker; the job must build its view and then
#           stop at the first CUDA call (exit != 0) with no row written anywhere (used by tests/test_root.sh)
#
# Guards for a real run (all must hold, else exit 1 before anything starts):
#   - the go marker $GO exists (written when the GPU queue gives this program the slot);
#   - nvidia-smi lists no compute process; the job then runs under gpu_run.sh EXCLUSIVE (timing runs: alone);
#   - cpu_load.py 10 shows at most 4.5 cores (the decision_layer queue gate; the sample is logged beside the time).
# The row gets the tag sb_<impl>_<tag> (only output names change: the tag is not read by the computation) and lands in
# ROOT/results/explore/. The job's reference model in the decision_layer abcd_models/ is hashed before and after, and the decision_layer record
# must hold no row or model with the bench tag afterwards; anything else is a FAILURE of the run (exit 3). Afterwards bench/fingerprint.py + bench/compare.py give the verdict against out/baseline/<JOBID>.json.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dl=${REPO_ROOT}/studies/decision_layer
py="$dl/.venv-dl/bin/python"
gpu_run="$dl/../tm_language_model/scripts/gpu_run.sh"
GO="${SPEED_GO:-${REPO_ROOT}/studies/decision_layer/speed/review/.gpu_go}"
root="${1:?usage}"; id="${2:?usage}"; impl="${3:?usage}"; dry="${4:-}"
root="$(cd "$root" && pwd)"
logd="$root/logs"; mkdir -p "$logd"
row=$(awk -F'\t' -v id="$id" '$1 == id' "$here/jobs.tsv")
[ -n "$row" ] || { echo "unknown job $id" >&2; exit 2; }
IFS=$'\t' read -r _ tag seed runner argv _ts _rec <<< "$row"
# 1. the script copies must still equal the decision_layer files
for f in abcd_explore.py table_run.py newtable_run.py; do
  a=$(sha256sum "$root/scripts/$f" | cut -c1-64); b=$(sha256sum "$dl/scripts/$f" | cut -c1-64)
  [ "$a" = "$b" ] || { echo "refused: $root/scripts/$f differs from the study's file" >&2; exit 1; }
done
# 2. the implementation
if [ "$impl" = "orig" ]; then pre=(); name=orig
else
  [ -f "$impl" ] || { echo "no patch file $impl" >&2; exit 2; }
  pre=("$here/with_patch.py" "$(realpath "$impl")"); name=$(basename "$impl" .py)
fi
btag="sb_${name}_${tag}"
case "$runner" in
  plain) cmd=(scripts/abcd_explore.py) ;;
  table:*) cmd=(scripts/table_run.py --table "${runner#table:}" scripts/abcd_explore.py) ;;
  newtable:*) r=${runner#newtable:}; cmd=(scripts/newtable_run.py --table-dir "data/tables/${r%%:*}" --name "${r#*:}" scripts/abcd_explore.py) ;;
  *) echo "unknown runner $runner" >&2; exit 2 ;;
esac
read -r -a args <<< "$argv"
envs=(HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 DL_GPU_FRAC=0.25 PYTHONDONTWRITEBYTECODE=1)
[ "$id" = "B6" ] && envs+=(TOKENIZERS_PARALLELISM=false)
log="$logd/${id}_${name}_$(date -u +%Y%m%dT%H%M%SZ).log"
# what this run could touch in the decision_layer record: the reference model of the job, and a row / model under the bench tag
refm="$dl/results/explore/abcd_models/${tag}_seed${seed}.npz"
snap() { sha256sum "$refm" | cut -c1-64; }
leak() { grep -c "\"tag\": \"$btag\"" "$dl/results/explore/abcd.jsonl"; find "$dl/results/explore/abcd_models" -maxdepth 1 -name "${btag}_seed${seed}.npz" | wc -l; }
before=$(snap)
if [ "$dry" = "--dry-cpu" ]; then
  echo "$(date -u +%FT%TZ) DRY-CPU $id $name: ${cmd[*]} ${args[*]} --seed $seed --tag $btag" >> "$log"
  (cd "$root" && env CUDA_VISIBLE_DEVICES= "${envs[@]}" nice -n 19 "$py" -B "${pre[@]}" "${cmd[@]}" "${args[@]}" --seed "$seed" --tag "$btag") >> "$log" 2>&1
  rc=$?
else
  [ -e "$GO" ] || { echo "refused: no go marker $GO (the GPU queue gives the slot)" >&2; exit 1; }
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
  [ -z "$apps" ] || { echo "refused: GPU compute processes running: $apps" >&2; exit 1; }
  m=$("$py" -I "$dl/scripts/cpu_load.py" 10); c=$(awk '{print $3}' <<< "$m")
  awk -v c="$c" 'BEGIN { exit !(c <= 4.5) }' || { echo "refused: $m" >&2; exit 1; }
  { echo "$m"; echo "$(date -u +%FT%TZ) start $id $name: ${cmd[*]} ${args[*]} --seed $seed --tag $btag"; } >> "$log"
  (cd "$root" && "$gpu_run" exclusive -- env "${envs[@]}" "$py" -B "${pre[@]}" "${cmd[@]}" "${args[@]}" --seed "$seed" --tag "$btag") >> "$log" 2>&1
  rc=$?
fi
echo "$(date -u +%FT%TZ) exit $rc" >> "$log"
after=$(snap); lk=$(leak | tr '\n' ' ')
if [ "$before" != "$after" ] || [ "$lk" != "0 0 " ]; then
  echo "FAILURE: the decision_layer record touched (reference model $before -> $after; rows / models with $btag: $lk)" | tee -a "$log" >&2; exit 3
fi
echo "the decision_layer record untouched: reference model ${refm##*/} $after; no row or model with tag $btag there" >> "$log"
echo "$log rc=$rc tag=$btag"
exit "$rc"
