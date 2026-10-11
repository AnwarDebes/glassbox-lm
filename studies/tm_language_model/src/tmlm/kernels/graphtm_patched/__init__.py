"""The E2 patch set: exact Binomial negative draw, sparse y_target, evaluate_sparse, sync removal, safe grid.

Normative text: docs/02; docs/03 Track A; tests/test_binomial_draw.py (M5).

Planned modules (written at the milestone named above; nothing here is implemented yet):
- kernels.py: starts as a byte-identical copy of the stock file (see README.md here)
- binomial_draw: host-computed Q64 CDF table, K_MAX=32
"""
