"""The three candidate configurations of FastCoTM (fast_cotm.py in this folder), by name. Nothing else lives here.

Keys = the switch keywords of FastCoTM.__init__. evalonce (tm/E1, rejected in LEDGER.md A2) and probs_expand
(tm/T2-expand, a separate later candidate) are OFF in every configuration. pace = 1 is the prototype's pacing value
(tmcode/BASELINE_tmcode.md: R1 "pace=1 in fast_tm.py"; GPU plan "B1 with pace 1"): after each training step the host
records a blocking-sync CUDA event and waits for the one recorded one step earlier (a no-op on the CPU).
"""

T1T3 = dict(xonly=True, probs=True, elem=True, probs_expand=False, evalonce=False, pace=0)
R1 = dict(xonly=False, probs=False, elem=False, probs_expand=False, evalonce=False, pace=1)
T1T3_R1 = dict(xonly=True, probs=True, elem=True, probs_expand=False, evalonce=False, pace=1)
