"""Receipts and the standalone checker (M8).

Normative text: docs/03 'Certification spec'; docs/05 E11; docs/07 G5; docs/13 H2.4.

- receipt: builds the receipt document of a generation (window, fired clauses with literals and
  votes, class sums, claimed token, decision mode, counter-based draw, calibration, state_hash,
  bank sha256; licensing and eliminating clauses for the elimination decoder) and the tampering
  modes of the acceptance test.
- checker_standalone.py: imports nothing from the project (standard library and numpy; z3
  optional); re-derives every token from the bank export and the receipt; exit 0 iff all pass.
"""
