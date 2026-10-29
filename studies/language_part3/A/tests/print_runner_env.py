"""Self-test helper, not a test: run through run_job.sh (its test mode) by test_listing_gate.py; prints the run-rule
environment the runner gave the job as one JSON line."""
import json
import os

print(json.dumps({**{k: os.environ.get(k) for k in ("A_RUNNER", "PART3_CHAIN_LOCK", "OMP_NUM_THREADS")},
                  "nice": os.nice(0)}))
