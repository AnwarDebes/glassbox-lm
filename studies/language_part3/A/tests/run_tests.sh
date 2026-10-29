#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"  # published copy: the repository root
# Self-tests of A-Struct: pytest in the counting venv (numpy, numba, sklearn), plain-assert TM tests in the torch venv,
# all as python -I -B with the job environment; then the folder listings before and after are compared.
# The log starts with the SHA-256 of every code file it tests.
#   tests/run_tests.sh [pytest -k expression]
set -u
source ${REPO_ROOT}/studies/language_part3/A/env.sh
cd "$A/tests" || exit 2
echo "self-tests start $(date -u +%FT%TZ); code under test:"
(cd "$A" && sha256sum astruct/*.py ./*.py ./*.sh tests/*.py tests/*.sh tests/pytest.ini)
before="$A/tmp/listing_tests_before.txt"
after="$A/tmp/listing_tests_after.txt"
"$PY_TM" -I -B "$A/listing.py" snapshot "$before" || exit 2
rc=0
if [ $# -gt 0 ]; then
  nice -n 19 "$PY_TM" -I -B -m pytest -c "$A/tests/pytest.ini" --rootdir "$A/tests" -q -k "$1" "$A/tests" || rc=1
else
  nice -n 19 "$PY_TM" -I -B -m pytest -c "$A/tests/pytest.ini" --rootdir "$A/tests" -q "$A/tests" || rc=1
  nice -n 19 "$PY_DL" -I -B "$A/tests/tm_tests.py" || rc=1
fi
"$PY_TM" -I -B "$A/listing.py" snapshot "$after" || exit 2
"$PY_TM" -I -B "$A/listing.py" compare "$before" "$after" || rc=1
echo "self-tests end $(date -u +%FT%TZ)"
exit $rc
