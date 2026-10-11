"""Samplers and decoders (M8).

Normative text: docs/03 'Diffusion sampler spec', 'Certification spec', 'Probabilities and
perplexity'; docs/04 M8; docs/05 E4 and CP; docs/13 H2.2, H6.4.

Modules:
- bank: the exported clause bank (`export_bank`) and the vectorised window evaluator (literal semantics of docs/11 section 1.4)
- scorer: `TrainedModel` and `WindowModel`, the C engine bound to a mutable generation window
- conftm: two-scalar calibration fitted on validation NLL, split-conformal sets, ECE
- sampling: counter-based draws (docs/11 section 3) shared by every decoder
- anyorder: any-order autoregressive sampler and the exact any-order NLL
- diffusion: masked-diffusion sampler (confidence-ordered unmasking, remasking) and the ELBO
- eliminate: three-valued elimination decoder with PySAT unit propagation
- incremental: incremental re-evaluation (T2.3), identical to full re-evaluation by construction and by test
- trace: the per-token records that become receipts (tmlm.certify)
"""
