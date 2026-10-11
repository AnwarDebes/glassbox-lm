"""Wrappers over GraphTsetlinMachine.Graphs and the masked-position graph builder (Track A, M5).

Normative text: docs/02 V1; docs/03 Track A; docs/04 T1.3.

Planned modules (written at the milestone named above; nothing here is implemented yet):
- build_masked_graphs(windows, hv_size=2048, hv_bits=8) -> Graphs, masked_position   # ONE mask per graph
- device_encoder: fill X/edge arrays directly and set the signature by hand (docs/02 blocker 3)
"""
