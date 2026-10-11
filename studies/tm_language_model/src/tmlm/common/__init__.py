"""Shared tooling: run logging, seeding, hashing, locks, and the model interface.

Nothing is imported here on purpose. `tmlm.common.runlog` must stay importable even
when numpy or the GPU stack is broken, because scripts/env_check.py uses it to report
exactly that situation.
"""
