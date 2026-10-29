# tm_language_model (the tmlm package): every target except help and bootstrap sources env/env.sh (PATH with
# .venv-tm/bin first, CUDA, thread caps = 6) and uses the project interpreter .venv-tm/bin/python. Only the
# target bootstrap installs anything (env/bootstrap.sh builds the environments and refuses to overwrite them).

SHELL := /bin/bash
.SHELLFLAGS := -euo pipefail -c
.DEFAULT_GOAL := help

ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
PY := $(ROOT)/.venv-tm/bin/python
ENV := source $(ROOT)/env/env.sh

# Extra flags for scripts/env_check.py, for example: make env-check ENV_CHECK_FLAGS=--no-gpu
ENV_CHECK_FLAGS ?=

.PHONY: help env-check lint test test-slow test-gpu smoke smoke-gpu bootstrap configs-check

help:
	@echo "targets:"
	@echo "  env-check  machine limits, GPU, imports, API signatures -> results/*_envcheck/"
	@echo "  lint       every measured number cites a log under results/"
	@echo "  test       pytest -q -m 'not gpu and not slow' tests, with every CUDA device hidden"
	@echo "  test-slow  the slow tests that need no GPU (-m 'slow and not gpu'), every CUDA device hidden"
	@echo "  test-gpu   the gpu-marked tests, under the shared GPU lock (scripts/gpu_run.sh)"
	@echo "  smoke      lint + test, then scripts/smoke_e2e.py: the study end to end on tiny data (CPU)"
	@echo "  smoke-gpu  scripts/smoke_e2e.py --gpu: adds CUDA = CPU state hash, under the shared GPU lock"
	@echo "  bootstrap  build both environments from scratch with a log (env/bootstrap.sh; refuses to overwrite)"
	@echo "  configs-check  validate experiments/configs/*.json against docs/05, docs/13 and EngineConfig"

# M0 acceptance: writes results/<timestamp>_envcheck/ with the cgroup limits and the
# printed signatures (protocol rule 10). Importing GraphTsetlinMachine.tm creates a CUDA
# context, so the check runs under the shared GPU lock.
env-check:
	$(ENV) && $(ROOT)/scripts/gpu_run.sh shared -- $(PY) $(ROOT)/scripts/env_check.py $(ENV_CHECK_FLAGS)

# Design decision, docs/12 E4: the logged recipe. Run from a clean checkout.
bootstrap:
	$(ROOT)/env/bootstrap.sh

configs-check:
	$(ENV) && $(PY) -m pytest -q $(ROOT)/tests/test_configs.py $(ROOT)/tests/test_tm_core_constants.py

lint:
	$(ENV) && $(PY) $(ROOT)/scripts/check_numbers_have_logs.py --root $(ROOT)

# 2026-10-25 (finish pass, TLM-B3): the CPU suite keeps off the GPU. The marker expression leaves out the
# gpu and slow tests, and CUDA_VISIBLE_DEVICES is empty, so no collection-time probe can create a CUDA
# context either (tests/conftest.py also hides the GPU unless the GPU lock is held). TEST_FLAGS adds options.
TEST_FLAGS ?=

test:
	$(ENV) && CUDA_VISIBLE_DEVICES= $(PY) -m pytest -q -m "not gpu and not slow" $(TEST_FLAGS) $(ROOT)/tests

# The slow tests that need no GPU (three since 2026-10-25: the data pipeline end to end on synthetic rows and
# the two KB1 parity tests against tmu; about 3 minutes on the CPU), left out of `test`.
test-slow:
	$(ENV) && CUDA_VISIBLE_DEVICES= $(PY) -m pytest -q -m "slow and not gpu" $(TEST_FLAGS) $(ROOT)/tests

# The gpu-marked tests (slow ones included) as correctness runs under the shared GPU lock; GPU jobs go
# through the queue of the machine, one at a time.
test-gpu:
	$(ENV) && $(ROOT)/scripts/gpu_run.sh shared -- $(PY) -m pytest -q -m gpu $(TEST_FLAGS) $(ROOT)/tests

# End-to-end smoke on tiny data (2026-10-25, finish pass, CR-15; it replaces the earlier placeholder):
# lint and the CPU tests, then scripts/smoke_e2e.py (env, CPU engine training with its golden state hash
# and determinism, learning on planted rules against the most frequent token, generation with receipts
# through the standalone checker, tampered receipts refused). smoke-gpu adds the step in which the CUDA
# engine reproduces the CPU engine's state hash, under the shared GPU lock. The Track A patched kernels
# of the placeholder are not part of it.
smoke: lint test
	$(ENV) && CUDA_VISIBLE_DEVICES= $(ROOT)/scripts/cpu_slot.sh 1 -- $(PY) $(ROOT)/scripts/smoke_e2e.py

smoke-gpu:
	$(ENV) && $(ROOT)/scripts/gpu_run.sh shared -- $(PY) $(ROOT)/scripts/smoke_e2e.py --gpu
