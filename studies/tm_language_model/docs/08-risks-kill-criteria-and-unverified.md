# 08 Risks, kill criteria, and unverified items

All kill criteria of the initial design are kept. KB1..KB7 are the new Track B criteria from the design decisions. The top-10 risks are from the design decisions. Milestones refer to docs/04.

## Kill criteria from the initial design (with thresholds and where they are read)
- Bigram not beaten on masked-token top-1 in Phase 0 (T0.4/E1, M1; the comparator is the left bigram): stop and diagnose.
- Class loop not dominant in the profile split (< ~40% of step time; T0.2, M1; measured with nvprof because nsys is absent): drop the sparse-draw headline; keep it as engineering.
- Sparse draw speedup < 2x at Q=1 (E2, M5): keep dense; drop the headline.
- Compositional head more than 3 points below flat even with learned codebooks (T1.4, M6): fall back to a label tree.
- OR port not landing by end of month 8: ship Specialist A for the C1 claim. Status: the kill-criterion path is taken from the start (docs/12). C1 is delivered by cluster literals and vote-set OR groups; Specialist A is an optional demo with a 1 day cap.
- Diffusion ELBO more than 2x the AR baseline after calibration (M8; read as ELBO perplexity vs the causal TM perplexity): pivot to any-order AR decoding (RADD); the headline then uses any-order AR decoding.
- Frozen transfer no gain (E5, M9): report the negative result with ablations.

## Track B kill criteria
- KB1 (M3): dense-equivalent mode > 2 points below tmu after 2 days of debugging: stop, the spec is wrong.
- KB2 (M3): tiny LM below the bigram: stop and diagnose. Beats it but < 70% of count-probe top-1: iterate R3-R6 up to 3 days, then fall back to count-initialized clauses refined by TM feedback, LABELLED as such.
- KB3 (M3, s3): B=1024 loses > 1 point vs B=1: default B = largest within 0.5 points; if < 64, micro-batch and re-derive throughput.
- KB4 (M7, s1): > 5,000 candidates per example at C=1M: pair anchors, then cap cluster-anchored clauses.
- KB5 (M7, s2): bags add < 0.3 points top-1 and < 2% perplexity at N=4096: report the negative result, keep the cost-vs-N curve.
- KB6 (M4): GPU engine not bit-exact with CPU after 3 days: the GPU becomes inference-only.
- KB7 (M7, CP): sparse-only causal perplexity > 1.5x KN-5: enable the wide tier [superseded by docs/12 F19]; still above KN-5: report perplexity as a first measurement with no superiority claim.

## Pre-registered targets and the 2026-09-22 additions
Every threshold above and every target of the three pillars is pre-registered in docs/13 with what a miss means. Additions of 2026-09-22 (docs/12 F):
- s5 (M3, M4): clause-sharded training not bit-exact with unsharded: docs/11 section 9 is incomplete; fix the spec, rerun; no multi-device statement until it passes. Never a claim without the passing log.
- CM (M4, M7): the fitted cost model is used only inside the measured range; an extrapolation is drawn dashed and labelled [derived] and never appears in a claim.
- H1, H2 (M9): the hierarchical row and the layer-2 delta are REPORTED criteria, not kill criteria: whichever direction the numbers go, the table is published with 5 seeds and the paired test of docs/13.
- TB (M7): the equal-cost transformer placement has no threshold; it is reported in both directions.
- Nothing is cut (docs/12 F4). The contingency order of docs/04 applies only on an explicit decision.

## Positive-by-construction check
Existence and proven-efficiency claims cannot come out negative; first-measurement claims cannot either. "Beat BERT" is never the thesis claim. Everything else in docs/07 is a target: a missed target is reported as a negative result with its log path.

## Top-10 risks and integrity rules
1. TM feedback cannot sustain rare patterns at V=16k (the frequency trap, docs/02). Mitigation: R4 lazy decay scoped to owned classes, the s sweep in E1, `count_probe`, the deviation ladder, KB2.
2. Batch staleness over-recruits clauses (no published analysis of batch-synchronous TA feedback). Mitigation: clause-sequential application, event and seed caps, experiment s3, KB3.
3. Scope explosion. Mitigation: reduced Track A, explicit deferrals, the cut order (docs/04), gates before scaling.
4. Environment: 6-CPU quota, 96 GiB of RAM with no swap, noisy neighbours, one 32 GB GPU. Mitigation: thread caps at 6, `mem_guard` at 85 GiB, repeated timings with variance, the GPU lock.
5. Contamination and leakage. Mitigation: report test/train n-gram overlap and de-overlapped results; BERT labelled "saw Wikipedia"; the from-scratch transformer is the clean neural comparator; subword vs whole-word results split; validation for ALL tuning; every test-set access counted; parser features from masked text only (docs/06).
6. Perplexity comparability. Mitigation: bits per byte, a labelled per-word conversion, KN-5 on identical tokens as the only like-for-like comparator (docs/05 CP).
7. GPU toolchain (CuPy + nvcc on sm_70 with a 12.8 driver). Mitigation: M0 microtests, no device atomics, one shared header for g++ and nvcc, PyCUDA fallback, KB6.
8. Energy claims. Mitigation: GPU joules measured under the exclusive lock, idle power subtracted and reported, both regimes; CPU energy [derived] unless RAPL is readable.
9. Cherry-picked generations or hollow receipts. Mitigation: pre-registered prompts, all samples logged, the standalone checker, tampered receipts must fail, no human-quality claims.
10. Overclaiming novelty and silent stock bugs. Mitigation: the qualified "first" claim with the citations of docs/01 and docs/07; the claims audit (`check_numbers_have_logs.py`); always pass the V100 grid; seed before importing GraphTM.
11. Time pressure. Fast results are a requirement, and a rushed run is a run that gets redone. Mitigation: time is saved only by ordering and parallel jobs (docs/04 "Lane schedule"; docs/12 F6 to F8), the smoke test, cached corpus arrays, checkpoint and resume, and never by dropping a measurement (docs/12 F4).

## Risks carried over from the initial design (status)
- "MultiOutputGraphTsetlinMachine may not exist in the pinned 0.3.3": RESOLVED. It exists in master [verified in source: tm.py:961] and we vendor master.
- "Constructor defaults for tm.py and TMAutoEncoder are [stated]": RESOLVED for the facts in docs/02, read from source; M0 still prints the signatures into `results/*_envcheck/`.
- spaCy neg attaches to the auxiliary, causing scope ambiguity. Mitigation: DependencyMatcher plus governor-subtree propagation; unit test on the slide-10 pair of the supervisor's project proposal (unpublished). Parser features are now an ablation only.
- New stock-package risks: the seven blockers of docs/02 (dense `encoded_Y`, one graph at a time, pure-Python `Graphs`, H100 default grid, import-time curand seeding, whole-dataset upload, `atomicAdd` under nvcc 12.9).

## Unverified list (do not present as fact)
- No training run has been logged yet; all model numbers are targets until logged. [superseded: training runs are logged from 2026-09-22 on; docs/results.md lists them with their run ids]
- `results/20260922T111529Z_envcheck/` was produced on a temporary A100 MIG slice (docs/12 F9) and is never cited.
- The HGTM and Multi-HGTM facts (docs/01) are [stated] from their repositories as read on 2026-09-22 and are not re-measured here; the layer-2 mechanism is re-measured (H2).
- The per-token cost bounds of docs/11 section 10 and the exchange list of docs/11 section 9 are design statements until s5 and CM log them.
- All Track B throughput and memory figures in docs/03 are [derived, unverified] until M3/M4 replace them. The Track A ceiling "near 3k examples/s" and the stock "20 to 150 examples/s" are [derived].
- Whether the stock GraphTM kernels (`atomicAdd`) compile under nvcc 12.9 (M0 microtest).
- Whether the default H100 grid actually overruns the curandState array on the V100 (we never use it).
- `Graphs` construction speed of 2k to 4k graphs/s (inferred, not measured).
- The launch count of 10 to 14 kernel launches per example [stated: design notes].
- That 0.3.3 and master kernels are byte-identical [stated: design notes].
- tmu `platform='CUDA'` broken on main and the unseeded C `rand()` on the CPU path [stated: design notes]; confirm in T0.5.
- RAPL readability (M0). Machine facts in the research protocol ("Hardware") are [stated] until `make env-check` logs them.
- HexDemo exact final testing accuracy (do not quote a specific figure until read from a run log; the ~99% claim could not be confirmed).
- GraphTM numbers: now read from v2's tables (docs/01). The initial design's "89.86 vs GCN 70.87" recommendation figure is NOT in v2 (v2 Table 7 has GraphTM 89.85 vs GCN 91.11 at noise 0.1) and is dropped; whether it came from v1 is unverified.
- arXiv ids: all 34 exist with the expected titles [verified in source, 2026-09-22: `results/20260922T125436Z_prior_art_check/C_reference_and_tool_verification.md`]. Still unverified: the NUMBERS quoted from rows labelled [stated] in docs/01; the RDDLGN Figure 3 column order (Table 2 is read: validation PPL 95 at 8k shared vocabulary, 209 baseline); whether the 2024 "Logical Transformer" paper ever appeared. Now read in full: Abdelwahab et al. ISTM 2025, Gorji et al. 2023, Zhang et al. ICLR 2026, GraphTM v2; GraphTM v2 is now read in full; only the recommendation figure differed (above).
- cair/PyHierarchicalTsetlinMachineCUDA main moved far past the vendored commit c85ca4e7 (HEAD 5b4a223 on 2026-09-22, version 0.3.2, branches `and_vote_normalization*` and `local_update_p*` that change the update rule). H1 must pin one commit, print its signatures into `results/`, and re-read `tm.py` before any run; the three MultiOutputTsetlinMachine defects are still present at that HEAD [verified in source: raw tm.py, 2026-09-22].
- UiA AURA thesis repository was unreachable from this container; the two 2026 master's theses named in search snippets (a wireless TM autoencoder; TM inference on mobile) are unverified and not on language.
- The environment recipe. The command block in the research protocol ("Environment setup") is a reconstruction from uv's output logs and the dist-info directories: the commands themselves were never logged and `env/bootstrap.sh` does not exist yet. Only the installed versions are measured (docs/02 V4). [superseded: `env/bootstrap.sh` exists and was run with its log on 2026-09-22, docs/12 E4; research protocol, "Environment setup"]
- The ISTM 2026 program is now read from istm.no (docs/01); paper contents are not.
- ICLR 2026 Zhang convergence paper.
- The two-phase TM-AE six-month DGX H100 figure (a projection).
- FNS full text; evaluate caching behaviour (inferred).
- The "216 variants" count of the supervisor's project proposal (unpublished; test empirically in G2).
- The earlier "BLiMP human 0.885" figure is corrected to 96.4% human aggregate and "GPT-2 8 points below" [verified in paper]; treat the old 0.885 as conflated.
- Numbers that do not exist in the literature and must be measured, never cited: BERT-base MLM top-1 on WikiText-103, KN-5 perplexity on WikiText-103, bidirectional n-gram top-1, BERT joules per token on a V100. The figure "KN about 150" (flagged in a design decision) is unsupported.
