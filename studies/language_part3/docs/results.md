# Results of the language_part3 study

Two registered experiments (both frozen 2026-10-21, before any real run) on Tsetlin machines that write text, each
against equally tuned counting systems. The numbers below are read from the files named beside them.

## A-Struct (R1): next-word choice for MultiWOZ system turns

**Stopped at its CPU gates, before any pilot or GPU run.** The gate stage (`A/out/s2d_gates/gates.json`) passed the
floor and twin checks (K8, and K9 on all nine families) but failed K3 on both co-primary arms, so neither arm went on to
its K3c pilot (`go_to_K3c` false for A-full and A-nf):

| Arm | Best learned family minus the counting system B (tune-hold bits per token; negative = better) | K3 |
|---|---|---|
| A-full (364 bits) | trees -0.0796 (part i passes), but part ii fails | not passed |
| A-nf (320 bits) | trees +0.0286 (worse than B) | not passed |

By the registration a K3 stop on both arms ends A's line; no TM was trained for A.

## C-core (R1): a TM class model inside KN5 + cache on PTB

**Passed its first gates; the GPU TM step never ran.** `C/data/step6_decision.json`: "PASS: K0 and K0b passed; the GPU TM
step may be scheduled". C-core asks whether a TM that predicts the next word's class (65 classes) from 60 dense code bits is as good as a
counted class model on the same codes, inside KN5 + cache. The gates before the TM: K0, the class model gains 0.146 bits
per word over the floor (needed 0.0439); K0b, the best learner (LightGBM) is within 0.0031 bits per word of the counted
twin (allowed 0.0072); the A/A check passes. The step-9 code
(the TM on the GPU) was written later, but its amendment was never frozen, so step 9 has not run.
