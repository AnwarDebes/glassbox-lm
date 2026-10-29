"""A-Struct v2 (Part 3, PLAN_part3.md section 3): code for S0 (data), S1 (counting floors, KenLM check K1),
S2 (rows, cross-fitted count bits, count twin, tree twin, LR1/LR2, boosted trees, A/A, shuffled labels, gates)
and the K3c CPU pilot of the Tsetlin machine. Imported code (tally floor/, dl tm.py, tokenize_ref.py) is used
read-only; every job runs as python -I -B with its caches and temporary files under A/tmp/.
"""
