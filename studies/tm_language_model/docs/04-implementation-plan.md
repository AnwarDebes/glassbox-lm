# 04 Implementation plan

The initial design's calendar phases (Phase 0 to Phase 4, months 1 to 9) are replaced by eleven gated milestones, M0..M10, from the design decisions. Every task id of the initial design (T0.1..T4.3, G1..G8, E0..E11) is kept and mapped to a milestone at the end of this file, or listed under "Deferred" with its reason. Durations and GPU-hours are [derived] estimates: total about 230 GPU-h, 3 to 4 weeks wall-clock. A milestone is done only when its ACCEPT line holds and the evidence is in a log directory under `results/`.

Update of 2026-09-22 (docs/12 F): the deliverable is the proof of concept of docs/07 (pillars P1, P2, P3 plus the proposal rows H1, H2), with thresholds pre-registered in docs/13. Nothing is cut (F4). Time is saved only by ordering and by the lane schedule below (F6 to F8). New items: pre-registration (M2 step 0), s5 clause-shard equivalence (M3, M4), CM cost model (M4, M7), TB equal-cost transformer (M7), H1 to H3 (M9).

## Lanes
ONE GPU lane (`scripts/gpu_run.sh`: shared lock for training and correctness, exclusive lock for timing, energy, and profiling; long runs checkpoint and yield on `.gpu_exclusive_request`), one CPU-heavy lane (`scripts/cpu_slot.sh`, 6 threads total), one light lane.

## M0 Bootstrap (about 0.5 day)
- Correct the hardware text (32 GB GPU, 6 CPUs, 96 GiB RAM).
- uv + both venvs. `env/env.sh` and `env/bootstrap.sh` exist; the bootstrap was run from a clean copy with its log on 2026-09-22 [measured: `results/_bootstrap/20260922T145118Z_bootstrap/`]; lock files in `env/`.
- `scripts/run_logged.py`, lints (`check_numbers_have_logs.py`), locks (`gpu_run.sh`, `cpu_slot.sh`), `mem_guard.py`.
- Microtests: pycuda on sm_70; CuPy sort/cumsum/RawModule via nvcc; integer `add.at` with duplicates; stock CAIR kernels compile (settles the `atomicAdd` question, docs/02 blocker 7); RAPL readability.
- ACCEPT: `make env-check` writes `results/*_envcheck/` with cgroup limits and printed signatures (protocol rule 10); lints pass.

## M1 Phase 0 on STOCK CAIR (1 to 1.5 days, 12 to 20 GPU-h; parallel with M2, M3)
- T0.1 / E0 `experiments/p0_scaling_harness.py`. Trivial two-node graph task; train MultiOutputGraphTsetlinMachine at outputs over 10..16000 ({10,100,1000,5000,16000}); record examples/s. First assert the class imports. Always pass `grid=(16*13,1,1)`. ACCEPT: a plot examples/s vs outputs saved to results/; slope characterised.
- T0.2 the `experiments/p0_*` profile script (the initial design's `p0_nsys_profile.sh`; nsys is absent, so it wraps nvprof). Profile one step with nvprof + CUDA events; parse the time split evaluate / select_clause_updates / update / update_message. ACCEPT: a table of the fractions in results/. KILL: class loop < 40% of step time: drop the sparse-draw headline.
- T0.3 `experiments/p0_compile_16000.py`. Instantiate with 16000 outputs; confirm compile. ACCEPT: no compile error logged.
- T0.4 / E1 `experiments/p0_masked_depth2.py`. ONE target per graph, V=2000, 5M tokens, full s sweep {10, 30, 100, 300, 1000, 3000} with `max_included_literals`, depth {1,2,3}, vs unigram / left bigram / right bigram / bidirectional n-gram. Staged grid (docs/05 E1, docs/12 F7): all s at depth 2 first, then depths 1 and 3 at the two best s values, the rest on disagreement. Runs under the shared GPU lock beside M3's CPU work. ACCEPT: masked-token top-1 beats the left bigram. KILL (as in the initial design): left bigram not beaten, stop and diagnose. Status 2026-09-22: not run yet; `experiments/p0_profile.py` (T0.2) exists with tests but has no log yet either.
- T0.5 `experiments/p0_tmu_ae_repro.py`. Reproduce TMAutoEncoder on a small corpus ON CPU (tmu CUDA is broken; the CPU path is unseeded, so the run is recorded as non-reproducible); report masked-token top-1/top-5. ACCEPT: numbers logged.
- T0.6: not used.
- T0.7 `tests/test_multioutput_hierarchy_typeerror.py`. Test that MultiOutputTsetlinMachine(hierarchy_structure=...) raises TypeError, documenting all three defects (docs/02 V3). ACCEPT: documents the known limitation.

## M2 Data + baselines (2 days CPU, about 30 GPU-h later)
- Step 0, before any run that feeds a reported number: `docs/13-preregistration.md` exists (done 2026-09-22) and `env/bootstrap.sh` has been run once from a clean copy with its output logged under `results/` (done 2026-09-22, docs/12 E4). M2 then writes `docs/notes/preregistration-data-hashes.json` once (tokenizer, cluster codes, eval mask sets, paragraph pairs) and never edits it.
- The paragraph-pair set of slide 11 of the supervisor's project proposal (unpublished) (E6 tier 1; H1, H2): pairs inside one split only, balanced, frozen, hashed.
- The small causal transformer script for TB (`.venv-dl`), ready so that it can run as soon as the causal TM run has a wall-clock and a joule figure.
- WikiText-103 raw + PTB with hashes; BPE 8k/16k on the train split only (the BPE part of T1.2); cluster codes; `corpus_store`; FROZEN eval mask sets with hashes (subword and whole-word variants); `contamination.py` (8-gram and 13-gram test/train overlap); all baselines of Track C (docs/06).
- ACCEPT: every baseline has a log dir; a test-set access counter exists.

## M3 Oracle + CPU engine (2 to 3 days, no GPU)
- `docs/11` normative text, `tm_core.h`, `oracle.py`, `engine.cpp`; dense-equivalent parity vs tmu; planted-rules synthetic corpus; tiny LM (5M tokens, V=2000) vs bigram and vs `count_probe` (what plain counting achieves in the same literal space); deviation ladder (s4); batch-staleness sweep (s3).
- s5 on the oracle: 2 and 4 shard instances with the exchange step of docs/11 section 9, hash-equal with the unsharded oracle. `DIAG_COUNTERS` carry candidates, verified literals, fired clauses and feedback events per token for CM.
- H2 row (d) at tiny-LM scale: layer 2 in tabular mode over layer-1 fired bits, delta over layer 1, 5 seeds (the M9 table reuses it).
- ACCEPT: oracle and C++ state hashes equal for 200 steps on 3 configs; NoisyXOR + one multi-class task within 2 points of tmu; >= 80% of planted rules recovered; s5 equal at 2 and 4 shards; measured CPU throughput replaces the estimates of docs/03. Gates KB1, KB2 (and KB3 from s3).

## M4 GPU engine (2 days, 6 GPU-h)
- s5 on the GPU (two engine instances on the one V100), CM constants at C in {8k, 64k, 1M} under the exclusive lock, checkpoint and resume so long runs yield to timing runs.
- ACCEPT: GPU state hash == CPU hash after 100 steps at B in {64, 1024, 4096}; hash invariant to launch configuration; s5 equal; `compute-sanitizer` clean; measured throughput table for C in {8k, 64k, 1M}, V in {8k, 16k}. Gate KB6. (Engine-level G6.)

## M5 Track A patches (1.5 to 2 days, 8 GPU-h; parallel with M3)
- T1.1 `src/tmlm/kernels/graphtm_patched/` (the initial design's `src/kernels/binomial_draw.py` + patched `kernels.py`). Implement the exact Binomial draw at the four sites (docs/02 pseudocode), sparse `y_target`, `evaluate_sparse`, sync removal.
- T1.3 `src/tmlm/graphs/masked_builder.py`. Masked-position graph builder, ONE mask per graph (not the initial design's multiple masks), vectorized builder and `device_encoder`. Accept: signal density log2(V) bits per masked position confirmed by counting; `test_graph_builder_equiv.py` (identical to stock `Graphs` on 1k random windows).
- E2 speedup per Q.
- ACCEPT: `test_binomial_draw.py` with the initial design's grid (M=200k, Q in {0.1,0.5,1}, CLASSES in {100,1000,16000}) and the CORRECTED assertions of docs/02, not the initial design's wording: the reference distribution is the exact Binomial(CLASSES-1, Q/(CLASSES-1)), NOT the initial design's Poisson(Q) (docs/12 C2: a Poisson reference rejects a correct sampler at CLASSES=100, Q=1), and assertions (1) and (2) count Q-gate selections with the error gate forced open, NOT realised updates (docs/12 B15); accuracy within seed noise of the dense path at E1 scale; speedup logged per Q. KILL: < 2x at Q=1: keep dense, drop the headline.

## M6 GATE 1 and the first causal run (3 to 4 days, about 100 GPU-h incl. Track A's 72 h)
- Order (docs/12 F6): first the CAUSAL training at V=8k with protocol CP (bits per byte vs the in-house KN-5, validation then test once), because TB, energy, receipts and generation all hang on that run; then E3 masked at V=8k, then V=16k. The Track A 72 h run starts as soon as M5 passes and yields to timing runs.
- E3: Track B masked model at V = 8k and 16k, C from 1M to 16M, validation then test; Track A row at V=8k; heads with the 3-point kill criterion (T1.4 `src/tmlm/heads/compositional_head.py`: K=8, M=32, cluster-code init, NN decode, fallbacks; accept: top-5 within 3 points of a flat one-hot head, else fallback); early negation smoke run (T3.4 says early).
- ACCEPT: Track B beats left bigram and CBOW-kNN (the initial design's Gate 1: beats bigram and kNN-over-GloVe; in-house CBOW replaces GloVe, docs/12); full WikiText-103 training wall time logged against the 48 h Gate 2 target; every number carries its log path. (G7.) Publication target ISTM 2027.

## M7 Scaling + causal perplexity (2 to 3 days, about 40 GPU-h)
- s1 accuracy and cost vs C from 64k to the largest count that fills 32 GB, log-log, 3 seeds at small sizes; V in {8k, 16k, 32k}; s2 CONTEXT N from 16 to 65536 at fixed clause budget with measured cost per token; CM constants and the fitted cost model over the s1 and s2 points; TB the equal-cost causal transformer (equal wall-clock and equal joules to the M6 causal run) plus the 24 GPU-h masked transformer; insertion-probability ablation (Track B's literal sampling, T2.2); wide-vote tier; pointer clauses.
- ACCEPT: causal perplexity and bits/byte vs in-house KN-5 at the reported size with 3 seeds; s1 and s2 figures with one line per seed; CM table with log paths; TB rows logged. Gates KB4, KB5, KB7; pre-registered H3.1 to H3.4.

## M8 Generation + certification (2 to 3 days, light GPU)
- T3.2 `src/tmlm/decode/diffusion.py` (MDLM sampler, confidence-ordered unmasking, remasking; ELBO logged); T3.3 / G1 `src/tmlm/decode/eliminate.py` (unit propagation over the exported bank with PySAT/z3; licensing and eliminating clauses per token); T2.3 `src/tmlm/decode/incremental.py` (per-step latency drop measured); T2.5 calibration (`src/tmlm/decode/conftm.py`: two-scalar fit + split-conformal stand-in); G5 `src/tmlm/certify/receipt_demo.py`; G6 `tests/test_deterministic_replay.py`; E11; Gates 2 and 3.
- ELBO kill criterion read as ELBO perplexity vs the causal TM perplexity (> 2x: headline uses any-order AR decoding).
- ACCEPT: 100% of receipts pass the standalone checker; tampered receipts fail; generated text and logs byte-identical on replay; ECE < 5%. Prompts are PRE-REGISTERED (first K test documents, seeds 0..k); every sample is logged.

## M9 Capability evals (2 to 3 days, about 20 GPU-h)
- T3.4 / E7 / G3 negation (minimal pairs, NaN-NLI, CondaQA once); E8 BLiMP; E5 transfer (the initial design's kill: report the negative result if no gain); E9 / G2 concept dictionary (`src/tmlm/eval/concept_dictionary.py`, WordNet purity, the "216 variants" test); E10 / G4 energy (`src/tmlm/eval/energy.py`); G8 / T2.1 crystallization (`src/tmlm/eval/crystallization.py`, absorbing thresholds 50/75/100, not 125); T1.5 / E6 coherence tiers 1 to 3 (tier 4 needs the external repository tm-llm-robustness-benchmark); polarity on/off ablation from masked text only (T1.2 featurizer).
- H1 the hierarchical package on the slide-11 pairs with flat-TM and logistic controls; H2 engine layer 1 and greedy layer 2 on the same pairs and on the LM at V=8k (rows a to d); H3 multi-view fusion, optional and last. E6 tier 1 is the H1 dataset.
- ACCEPT: every table of docs/06 has a log dir; the proposal table (H1, H2) has 5 seeds per row and the paired test of docs/13; negative outcomes are reported as such.

## M10 Report (1 to 2 days)
- T4.1: all tables and figures regenerated from `results/`; claims audit with `scripts/check_numbers_have_logs.py`; limitations and deviations; one reproduction command.

## Gates (the initial design's three gates, kept)
- Gate 1 (M6): WikiText-103 masked-token top-1/top-5 at 8k to 16k vocab beats the left bigram and CBOW-kNN.
- Gate 2 (M6 wall time, M8 calibration): full WikiText-103 training within 48 hours; ECE below 5%.
- Gate 3 (M8): the model generates with receipts that pass the standalone checker.

## Deferred, with reasons (also recorded in docs/12)
- T2.4 resonator decoder: exact nearest-neighbour decode is trivial at V=16k.
- T3.1 OR port into GraphTM layer zero: take the initial design's kill-criterion path; C1 is delivered by cluster literals and vote-set OR groups; Specialist A demo optional, 1 day cap.
- T4.3 theory chapters (BMF/FCA clause-count theorem, Hopfield capacity fit): not part of the measured milestones; M7 supplies the empirical capacity curve.
- T4.2 d-DNNF compilation: optional, toy bank only.
- Pretrained GloVe-kNN baseline: word vectors do not align with BPE targets and are contaminated; the in-house CBOW replaces it.
- Depth 1/2/3 ablation: Track A only (E1); Track B has no depth.
- From Track A only (docs/03): batched GraphTM training, absorbing states, literal streaming, incremental re-evaluation, specialist composition, heads. Reason: launch-bound ceiling near 3k examples/s [derived]; message passing buys little when offsets are fixed. Their Track B counterparts are in M7, M8, and M9.

## Lane schedule (2026-09-22; durations [derived, unverified])
The engine is the critical path; everything measured waits on it. Three lanes, GPU use serialised by `scripts/gpu_run.sh`.
| Days | Lane 1: engine (critical path) | Lane 2: data, baselines, Track A | Lane 3: docs, tooling, evals |
|---|---|---|---|
| 1 to 5 | M3: oracle tests, C++ engine, parity vs tmu, planted rules, tiny LM, s3, s4, s5 (oracle), H2 row (d) | M2: bootstrap.sh from a clean checkout (logged), data, BPE, clusters, eval masks, pairs, KN-5, n-grams, CBOW-kNN, count_probe; M1 leftovers: E1 staged grid and the nvprof split on the GPU | pre-registration data hashes; TB script; receipt checker skeleton; CHANGELOG |
| 6 to 10 | M4: CUDA engine bit-exact, s5 (GPU), CM constants; M6: causal V=8k run and CP, then E3 masked V=8k | M5: Track A patches, E2, then the 72 h Track A E3 run under the shared lock | M8 decoders and receipts against the CPU engine |
| 11 to 15 | M7: s1, s2, CM fit, KB4/KB5/KB7; E3 at V=16k | TB at equal wall-clock and joules; energy E10 | M8 calibration, replay; M9 H1, H2, E9 |
| 16 to 20 | M9 remaining evals (negation, BLiMP, transfer, crystallization, coherence tiers 2 and 3, polarity ablation), H3 last | V=32k row; PTB | M10 report, claims audit, one reproduction command |
Long GPU runs checkpoint and yield on `.gpu_exclusive_request`; timing, energy and profiling always hold the exclusive lock. The time-saving mechanisms are B1 to B10 (fixed-token scaling points, sweep small and confirm large, epoch budget from the tiny LM, one anchor run for three tables, concurrent launch-bound jobs, vectorised counting, cached arrays, checkpoint and resume, cheap gates first, caps read as caps).

## Contingency order (applies ONLY on an explicit decision; nothing is cut as of 2026-09-22, docs/12 F4)
d-DNNF, coherence tiers 2 and 3, Specialist A, CondaQA, PTB, extra head variants, V=32k, top of the context range (65k down to 4k), Track A E3 run (72 h down to 12 h), GPU training (the GPU engine becomes inference-only).

## Mapping from task ids of the initial design to milestones
| Initial design id | What | Milestone | Note |
|---|---|---|---|
| T0.1, E0 | output-count scaling harness | M1 | safe grid always |
| T0.2 | profile split | M1 | nvprof, nsys is absent |
| T0.3 | compile at 16000 outputs | M1 | stock-kernel compile first checked in M0 |
| T0.4, E1 | masked sanity vs bigram | M1 | one target per graph, s sweep, depth {1,2,3} |
| T0.5 | tmu TM-AE reproduction | M1 | CPU only, non-reproducible |
| T0.6 | not used | M1 | not used |
| T0.7 | hierarchy TypeError test | M1 | three defects |
| T1.1, E2 | Binomial draw, equivalence, speedup | M5 | exact Binomial reference (not Poisson), four sites, test per docs/02 |
| T1.2 | featurizer | M2 (BPE), M9 (polarity ablation) | parser features from masked text only |
| T1.3 | masked graph builder | M5 | one mask per graph |
| T1.4 | compositional head | M6 | 3-point kill criterion kept |
| T1.5, E6 | coherence head | M9 | tiers 1 to 3; second in the cut order |
| Gate 1, E3, G7 | WikiText-103 masked top-1/top-5 | M6 | Track B main, one Track A row; E0 curve from M1 |
| T2.1, G8 | absorbing states, crystallization | M9 | thresholds 50/75/100 |
| T2.2 | literal sampling and streaming | M7 | as Track B insertion-probability ablation |
| T2.3 | incremental re-evaluation | M8 | |
| T2.4 | resonator decoder | deferred | NN decode is exact and cheap |
| T2.5 | ConfTM calibration | M8 | two-scalar fit + split-conformal stand-in |
| Gate 2, E4 | full training within 48 h, ECE < 5% | M6 (wall time), M8 (ECE) | |
| T3.1 | OR port | deferred | kill-criterion path |
| T3.2 | diffusion sampler | M8 | |
| T3.3, G1 | elimination decoder | M8 | |
| T3.4, E7, G3 | negation benchmark | M6 (smoke), M9 (full) | |
| Gate 3 | generation with receipts | M8 | |
| T4.1 | full evaluation suite | M9, M10 | |
| T4.2 | d-DNNF | deferred | optional, toy bank |
| T4.3 | theory chapters | deferred | not part of the measured milestones |
| E5 | frozen clause-code transfer | M9 | |
| E8 | BLiMP | M9 | pseudo-log-likelihood |
| E9, G2 | concept dictionary, synonym purity | M9 | |
| E10, G4 | energy per token | M9 | NVML, exclusive lock |
| E11, G5 | receipts | M8 | standalone checker |
| E11, G6 | deterministic replay | M4 (engine), M8 (generation) | |
| s1, s2 | clause and context scaling (new) | M7 | docs/05 |
| s3, s4 | batch staleness, deviation ladder (new) | M3 | docs/05 |
| s5 | clause-shard equivalence (new, 2026-09-22) | M3 (oracle), M4 (GPU) | docs/05, docs/11 section 9 |
| CM | cost model with measured constants (new, 2026-09-22) | M4, M7 | docs/05, docs/11 section 10 |
| TB | equal-cost causal transformer (new, 2026-09-22) | M7 | docs/05 |
| H1 | hierarchical package on the slide-11 task (new, 2026-09-22) | M9 | docs/05; replaces the optional Specialist A demo |
| H2 | engine layer 1 and greedy layer 2 (new, 2026-09-22; E5 promoted) | M3 (tiny LM), M9 | docs/05 |
| H3 | multi-view fusion ablation (new, 2026-09-22, optional, last) | M9 | docs/05 |
| pre-reg | docs/13 and `prereg_sha256` in every run.json (new, 2026-09-22) | M2 step 0 | docs/13; protocol rule 16 |
| TK | tokenization-level ablation, BPE vs word-level at tiny-LM scale (new, 2026-09-22) | M3 | docs/05 |
| E7 extension | counterfactual IMDb robustness ratio and TextAttack recipes through the harness of tm-llm-robustness-benchmark (2026-09-22) | M9 | docs/05 E7; `third_party/ref/tm-llm-robustness-benchmark` |
