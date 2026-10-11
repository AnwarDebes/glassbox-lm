"""Count-Vote Tsetlin Machine (CVTM), docs/11 section 11 (normative since 2026-09-24).

Modules (import them directly; this file holds no code):

- `tmlm.cvtm.kn_gpu`: the Kneser-Ney count fit of section 11.2 on the GPU (CuPy) or the CPU,
  array-identical to `tmlm.eval.baselines.ngram.KNModel.fit`, plus the count pruning used by
  docs/13 A3 H8.9.
- `tmlm.cvtm.suffix`: family S, the suffix clauses of section 11.2 (the exact vote form of
  Kneser-Ney: bias, clause constants ln gamma, follower votes; float64 and Q16 int32 forms,
  fired lists, explanations, state hash).
- `tmlm.cvtm.doc_clauses`: family D, the document clauses of section 11.3 (exact batch and
  streaming counts, the pointer clause, candidate successor rows).
- `tmlm.cvtm.mixture`: the fixed combination of section 11.4 (buckets, MAP-EM, gold probability,
  full distribution, residual votes).
- Sections 11.5 (count-vote read-out of a Tsetlin-learned bank, family L) and 11.8 (the masked
  variant) live in modules of their own; each module's docstring names the
  section it implements.
"""
