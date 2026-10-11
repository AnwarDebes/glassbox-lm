# tm_language_model (the tmlm package): every target sources env/env.sh (PATH, CUDA, thread caps = 6) and uses the
# project interpreter .venv-tm/bin/python. Nothing here installs anything.

SHELL := /bin/bash
.SHELLFLAGS := -euo pipefail -c
.DEFAULT_GOAL := help

ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
PY := $(ROOT)/.venv-tm/bin/python
ENV := source $(ROOT)/env/env.sh

# Extra flags for scripts/env_check.py, for example: make env-check ENV_CHECK_FLAGS=--no-gpu
ENV_CHECK_FLAGS ?=

.PHONY: help env-check lint test smoke bootstrap configs-check

help:
	@echo "targets:"
	@echo "  env-check  machine limits, GPU, imports, API signatures -> results/*_envcheck/"
	@echo "  lint       every measured number cites a log under results/"
	@echo "  test       pytest -q tests"
	@echo "  smoke      lint + test for now (see the comment in the Makefile)"
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

test:
	$(ENV) && $(PY) -m pytest -q $(ROOT)/tests

# Placeholder. The real smoke run (design decision, under 10 minutes) will add, in order:
#   1. env check
#   2. build the CPU engine
#   3. train 60 s on a 2M-token slice at V=2000, assert top-1 beats the unigram baseline
#      and the state hash equals the golden hash
#   4. 20 GPU steps bit-exact with the CPU engine
#   5. generate 50 tokens with receipts and pass them through the standalone checker
#   6. Track A patched kernels for 200 examples at CLASSES=100
#   7. lints
smoke: lint test
