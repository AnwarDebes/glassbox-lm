# graphtm_patched (Track A, milestone M5)

`kernels.py` here starts as a BYTE-IDENTICAL copy of the stock vendored file
`third_party/GraphTsetlinMachine/GraphTsetlinMachine/kernels.py` (master c6129e14), copied on
2026-09-22, sha256 `9d97a1fb3a586beda467d9749b41d9a3f35bf42679450bfedc55081f7860332a` [verified in source: `sha256sum` of both files].
The stock file is the reference and is never edited (research protocol, "Environment setup").

Patches to be applied in M5, one per commit of the CHANGELOG, each with a test (protocol rule 7):
1. sparse `y_target[N]` instead of the dense `encoded_Y` (docs/02 blocker 1);
2. `draw_negatives`: exact Binomial(CLASSES-1, Q/(CLASSES-1)) by inversion against a host Q64 CDF
   table, K_MAX=32, `class_needed` flags (docs/02 "Kernel edit points"; the four sites);
3. `evaluate_sparse`: one thread per needed class over compacted fired clauses, no atomics;
4. `select` / `update` / `update_message` over `1 + neg_cnt` classes, dense path behind `NEG_DRAW=dense`;
5. removal of the per-launch synchronisation; `device_encoder`; safe grid `(16*13,1,1)`.
Acceptance: `tests/test_binomial_draw.py` with the corrected assertions of docs/02 (exact Binomial
reference, Q-gate selections counted with the error gate forced open; docs/12 C2, B15).
