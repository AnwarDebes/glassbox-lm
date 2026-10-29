#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # published copy: the repository root
# slot2_next.sh: job slot 2 after T4 lane B (slot2_chain.sh, pid 1877406) ends: the registered TM-AE training of
# REGISTRATION_T5.md (CPU, one thread, nice 10), then the R1 / R3a val runs (t5_queue.sh tmae)
T5=${REPO_ROOT}/studies/text_understanding/t5
PYTM=${REPO_ROOT}/studies/tm_language_model/.venv-tm/bin/python
while [ -e /proc/1877406 ]; do sleep 60; done
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [slot2] T4 lane B gone; start train_tmae.py" >> "$T5/runs_t5.log"
[ "$(sha256sum "$T5/train_tmae.py" | cut -c1-16)" = 9c0ca84d0cef7373 ] || { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [slot2] Error: train_tmae.py hash changed" >> "$T5/runs_t5.log"; exit 1; }
[ "$(sha256sum "$T5/tmae_fixed.py" | cut -c1-16)" = bb179b9f2562524d ] || { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [slot2] Error: tmae_fixed.py hash changed" >> "$T5/runs_t5.log"; exit 1; }
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 nice -n 10 "$PYTM" -I "$T5/train_tmae.py" >> "$T5/runs_tmae.out" 2>&1
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [slot2] train_tmae.py exit $?; start t5_queue.sh tmae" >> "$T5/runs_t5.log"
bash "$T5/t5_queue.sh" tmae
