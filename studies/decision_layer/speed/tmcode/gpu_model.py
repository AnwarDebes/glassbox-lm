"""Arithmetic model of one CoTM training step on the V100 (no GPU used): bytes moved per (clause x literal), GEMM FLOPs,
kernel launches, calibrated on the recorded step times (steps_summary.tsv). Pure arithmetic; prints a table.

Byte counts per element of the [N, m] automaton block, read off dl/tm.py line by line (fp16 = 2 B, fp32 / int32 = 4 B,
bool = 1 B; a binary op reads two operands and writes one):
  feedback, every step:   incl = (ta > n).to(fp16)          read 4 + write 1, read 1 + write 2   =  8
                          co GEMM reads incl (fp16)                                               =  2
  _automata_chunked, per chunk element (the path of every e72+ run: N*m > CHUNK_ELEMS = 50M):
    g1^T L          GEMM out fp16 2, .float() 2+4                                                 =  8
    g1^T (1-L)      GEMM out 2, .float() 6, + nf 8                                                = 16
    g2^T (1-L)      GEMM out 2, .float() 6                                                        =  8
    excluded        ta <= n: read 4, write 1                                                      =  5
    full_like x2    write 4 each                                                                  =  8
    binomial x2     read count 4 + prob 4, write 4, each                                          = 24
    inc - dec 12; excluded.float() 5; cnt_t2 * that 12; + t2 12; .to(int32) 8; ta + 12;
    clamp_ 8; self.ta[a:b] = new 8                                                                = 77
  original total: 10 + 146 = 156 B per element per step.
Prototype T1+T2+T3 (fast_tm.py; counts from X only, cached probabilities, fewer passes), per element:
    A = g1^T X, B = g2^T X (half width) out 1 + 1; cnt_inc, cnt_dec, cnt_t2 from A / B with out= halves: 3 x 6;
    masked_fill of cnt_t2: 9; probabilities: 0; binomial x2: 24; inc.sub_(dec) 12; inc.add_(t2) 12;
    .to(int32) 8; ta.add_ 12; clamp_ 8                                                            = 105 (+ 10) = 115
  with an expanded (stride-0) probability: binomial reads 4 B less each                          = 107
Fused elementwise (T4, e.g. Triton through torch.compile or jiterator; same torch.binomial calls):
    counts kernel: read A, B (1 + 1), incl (1), write cnt_inc, cnt_dec (4 + 4) = 11; binomial x2 (expanded prob) 16;
    update kernel: read inc 4, dec 4, B 1, incl 1, ta 4, write ta 4, incl fp16 2 + bool 1 = 21; co GEMM 2   = 50
"""
from __future__ import annotations

BYTES = {"original": 156, "T1-T3": 115, "T1-T3 expand": 107, "T4 fused": 50}
GEMM_UNITS = {"original": 4, "T1-T3": 2, "T1-T3 expand": 2, "T4 fused": 2}  # units of 2*b*m*N FLOPs (b = 128)
# recorded alone-on-the-GPU step times, ns per (clause x literal) per step (steps_summary.tsv, ABCD Bc4 c4000):
REC = {8: 0.323, 12: 0.298, 16: 0.288, 24: 0.276}
# linear fit in 1/s through s = 8 and s = 24: intercept = the s-independent part, slope = the binomial's s-dependent part
slope = (REC[8] - REC[24]) / (1 / 8 - 1 / 24)
base = REC[24] - slope / 24
print(f"fit: t(s) = {base:.3f} + {slope:.3f} / s  ns per element-step (ABCD 4k, view B on C4)")
for bw in (600, 700, 850):  # GB/s effective
    gemm_ns = 2 * 128 * 1 / 80e12 * 1e9  # one unit, per element, at 80 TFLOPS fp16
    mem_orig = BYTES["original"] / bw  # ns per element (B / (GB/s) = ns)
    fixed_other = base - mem_orig - GEMM_UNITS["original"] * gemm_ns  # binomial base compute, launches, the rest
    line = [f"BW {bw} GB/s: original memory {mem_orig:.3f} ns, GEMM {GEMM_UNITS['original'] * gemm_ns:.3f}, other fixed {fixed_other:+.3f}"]
    for s in (8, 16):
        t0 = base + slope / s
        for k in ("T1-T3", "T1-T3 expand", "T4 fused"):
            t = t0 - (BYTES["original"] - BYTES[k]) / bw - (GEMM_UNITS["original"] - GEMM_UNITS[k]) * gemm_ns
            line.append(f"s{s} {k}: {t:.3f} ns ({100 * (1 - t / t0):.0f}% less)")
    print("\n  ".join(line))

# kernel launches per step (default path, chunked with k chunks), counted from dl/tm.py: 62 outside the chunks
# (literals 3, include mask 2, co 3, class sums 3, negative class 8, margins 8, weights gather 2, selections 4,
# feedback masks 14, weight update 6+, batch gather 2, notL 1, ...) and 27 per chunk; see BASELINE_tmcode.md
for k in (1, 2, 5):
    print(f"chunks {k}: about {62 + 27 * k} kernel launches per step")
for name, N, bits, s, ms in (("ABCD 4k s16 (e74c)", 4000, 6684, 16, 15.38), ("MultiWOZ 16k s8 (e77)", 16000, 6580, 8, 79.2),
                             ("STAR 16k s16 (e74c)", 16000, 7227, 16, 69.9)):
    Nm = N * 2 * bits
    print(f"{name}: N*m {Nm / 1e6:.0f}M, original {BYTES['original'] * Nm / 1e9:.1f} GB per step "
          f"({BYTES['original'] * Nm / 1e9 / (ms / 1e3):.0f} GB/s at the recorded {ms} ms/step); "
          f"T1-T3 {BYTES['T1-T3'] * Nm / 1e9:.1f} GB; T4 {BYTES['T4 fused'] * Nm / 1e9:.1f} GB")
