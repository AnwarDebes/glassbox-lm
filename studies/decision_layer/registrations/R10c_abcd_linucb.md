# R10c registration: the TM bandit against LinUCB on the ABCD workflow

R10 (met) left the comparison with LinUCB out, because no LinUCB run had finished on dev when it was frozen. One has now: on dev (seed 1, 15 passes, the same 1,706 label-free bits, alpha 0.5, `e28_linucbfast_a0.5`) LinUCB ends at 0.7725 (its best pass 0.7983; online 0.813), against 0.8350 for the TM bandit (online 0.721). R10c asks whether that holds on test, and whether it holds against every LinUCB exploration weight tried, not only the one tried on dev.

**What was known at the freeze:** R10's test results for the TM bandit (5 seeds, mean 0.8300) and the supervised TM (0.8404). No LinUCB result on test. The dev runs of LinUCB with alpha 0.1 and 1.0 were stopped unfinished (amendment).

Frozen by `scripts/freeze.py` before any R10c run. Amendments go in a dated file.

**Publication note:** the wording of this copy was edited for publication on 2026-10-23. The registered question, rules, thresholds, seeds, analysis plans, amendments, dates and times are unchanged; the freeze record next to this file holds the SHA-256 of the text as registered.

## Arms (test; `scripts/abcd_bandit.py --learner linucb --eval test --out confirm/R10`)

- **LinUCB** (`LinUCBFast`: disjoint LinUCB, ridge 1 per arm, Woodbury updates, same estimator as `LinUCB`; `tests/test_linucb_fast.py`) on the same 1,706 bits as the TM bandit, the same stream (15 passes, rounds of 32, the stream seed = the run seed), alpha in {0.1, 0.5, 1.0}, 3 seeds each (1, 2, 3).
- The TM bandit: R10's 5 runs, unchanged (no new TM run).

## Statistic

Per test action turn, each arm's correctness averaged over its seeds; paired differences; 95% interval from a bootstrap over the 1,004 test conversations, 10,000 resamples, seed 0 (`scripts/report_abcd_bandit.py`).

## Pass rules (test)

- **R10c.1 (the TM bandit beats LinUCB at the dev setting):** TM bandit minus LinUCB alpha 0.5 is at least +2.0 points, with the interval above 0.
- **R10c.2 (it beats every LinUCB setting tried):** TM bandit minus LinUCB at its best alpha on test (the best of the three by mean test accuracy; chosen on test, which favours LinUCB) has its interval above 0.
- **R10c is met** if R10c.1 and R10c.2 both hold.

## Reported with no rule

Online accuracy of each LinUCB arm (the share of reward 1 over the stream: LinUCB is expected to be higher than the TM's 0.718), LinUCB's best pass on test (chosen on test, an upper bound), AST-weighted accuracies.
