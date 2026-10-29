#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"  # published copy: the repository root
# step1a_chain.sh: logic-reader step 1a, EXPLORATORY, val only: concept cuts of the bert-base word vectors + oracle LR
here=${REPO_ROOT}/studies/text_understanding
py=${REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python
log=$here/runs_headroom.log
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
for spec in "256 1" "256 4" "1024 1"; do
  set -- $spec
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) start step1_concepts.py --K $1 --pq $2 (step1_concepts.py $(sha256sum $here/step1_concepts.py | cut -c1-16))" >> "$log"
  nice -n 5 "$py" -I -B "$here/step1_concepts.py" --emb-dir "$here/emb" --teacher bert-base --K "$1" --pq "$2" >> "$log" 2>&1
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) end step1_concepts.py --K $1 --pq $2 exit $?" >> "$log"
done
for name in bert-base_K256 bert-base_K256_pq4 bert-base_K1024; do
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) start headroom_lr.py --onehot $name (headroom_lr.py $(sha256sum $here/headroom_lr.py | cut -c1-16))" >> "$log"
  nice -n 5 "$py" -I -B "$here/headroom_lr.py" --emb-dir "$here/emb" --teacher bert-base --onehot "$name" --extra minilm3 >> "$log" 2>&1
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) end headroom_lr.py --onehot $name exit $?" >> "$log"
done
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) step 1a chain done" >> "$log"
