# 03 Architecture

This file is rewritten around the design decisions: one interface, three tracks. The initial design's single GraphTM pipeline survives as Track A, reduced to what the evidence in docs/02 supports. The main model is the Track B sparse engine. The initial design's component specs are kept below where they are still valid. Every departure is listed in docs/12.

## Thesis claim (the reason this is tractable)
TM-AE cost is linear in vocabulary because it rebuilds the input once per target word [verified in source: docs/02 V2]. The initial design removed that with three changes: (1) a shared masked-position input supplies a signal for the whole vocabulary from one clause evaluation; (2) the exact Binomial draw removes the class loop losslessly; (3) compositional codes remove the vocabulary from the output layer. Source reading (docs/02, "Blockers in the stock package") shows that on stock GraphTM these three are not enough: training is one graph at a time from Python with a dense target matrix, which caps even the patched path near 3k examples/s [derived]. Track B therefore adds a fourth change: (4) event-driven evaluation over an anchor index, with sparse signed votes. Cost per token is then set by the candidate clauses the index enumerates (kernels K2 and K3) and by the clauses that fire. What is claimed, and what is not [derived, unverified]:
- Context length N: cost per token is designed to be nearly independent of N (the context-cost rule below). This is a target, measured in experiment s2, not a result.
- Clause count C: cost is NOT independent of C. It grows with candidates per example and therefore with clause count: s1 measures it, kill criterion KB4 exists because of it (> 5,000 candidates per example at C=1M), and the estimates in the budget table at the end of this file fall from 50k to 150k examples/s at C=1M to 15k to 30k examples/s at C=16M.
- Vocabulary V: sparse votes remove the dense V x C weight matrix and the per-class feedback loop (R2 and R3 touch the target and 2 negatives). That does not make every step free of V, so cost vs V is measured in s1, not assumed.

Everything fits ONE 32 GB V100. By RADD, a per-position masked predictor is a complete masked-diffusion language model [verified in paper: RADD].

Design priorities, in order: (1) training scalability, (2) a very large context window at low cost, (3) keeping TM virtues: cheap bitwise compute, low resources, interpretability. Since 2026-09-22 the deliverable is stated as three pillars with pre-registered thresholds (docs/07 "Proof-of-concept criteria", docs/13): accuracy, reproducibility and extensibility, scaling with compute; the scalability proof rests on the per-token cost bounds of docs/11 section 10, the log-log curves of s1 and s2, the clause-shard equivalence of docs/11 section 9 (s5), and the cost model CM.

## One interface (`src/tmlm/common/interface.py`)
Every model (Track A, Track B, every baseline) implements:
```
predict_scores(batch)   -> integer class scores
logprobs(batch)         -> calibrated log-probabilities
explain(position)       -> fired clauses with literals and votes
state_hash()            -> hash of the full learnable state, for replay tests
export_bank()           -> clause bank in a format a standalone checker can read
```
`export_bank()` exists so that receipts are checkable by a standalone program that imports nothing from the project. Track C (baselines, heads, decoders, certification, evals) is written once against this interface.

## Track A: GraphTM, reduced to what the evidence supports
Stock CAIR GraphTM stays the Phase 0 measurement target and the reference for equivalence tests.

Keep:
- Stock Phase 0 measurements: E0 scaling curve, profile split, 16000-class compile, E1.
- The E2 patch set in a vendored fork (`src/tmlm/kernels/graphtm_patched/`):
  - sparse `y_target[N]` instead of the dense `encoded_Y`;
  - a `draw_negatives` kernel doing an EXACT Binomial(CLASSES-1, Q/(CLASSES-1)) draw by inversion against a host-computed Q64 CDF table, K_MAX=32 distinct negatives per clause, with `class_needed` flags;
  - `evaluate_sparse`: one thread per needed class over compacted fired clauses, no atomics;
  - `select` / `update` / `update_message` loop over `1 + neg_cnt` classes;
  - the dense path kept behind `NEG_DRAW=dense`;
  - removal of the per-launch synchronisation;
  - a `device_encoder` writing `X` from token windows on the GPU with one shared BANDED signed-distance edge template;
  - the safe grid `(16*13,1,1)` and explicit seeding before import.
- The initial design's distributional-equivalence test, with the corrected assertions of docs/02: exact Binomial reference instead of Poisson(Q), and Q-gate selections counted with the error gate forced open instead of realised updates (docs/12 C2, B15).
- ONE target per graph (GraphTM classifies graphs, not nodes: docs/02).
- ONE time-boxed E3-scale run (max 72 GPU-h) for a "GraphTM row" in the results table.

Drop from Track A: batched GraphTM training, absorbing states, literal streaming, incremental re-evaluation, the OR port, specialist composition, heads. Reason: a launch-bound ceiling near 3k examples/s [derived], and message passing buys little when offsets are fixed.

Exact Binomial, not Poisson: at CLASSES=100 the Poisson approximation differs by about 1% in variance, which the initial design's own 200k-trial chi-square can flag [derived].

Track A data flow (the initial design's pipeline, corrected):
```
raw text
  -> BPE (8k..16k)  -> token ids
  -> window builder -> for each window: nodes = tokens, ONE position replaced by MASK
        edges: signed distance (span 8), one shared banded template
        node symbols: tok_i, MASK (polarity symbols and dependency edges are an ablation, from MASKED text only)
  -> device_encoder (or Graphs.encode() at Phase 0 scale)  (hypervector codebook, size 2048, bits 8)
  -> patched MultiClassGraphTsetlinMachine  (depth 1..3, clauses 8192, message_size 1024, grid=(16*13,1,1))
        |-> class_sum for {target} union {drawn negatives} (Binomial-sparse)
        |-> transform() -> (clause outputs, class sums)
  -> the common interface -> Track C
```

## Track B: sparse event-driven TM engine (the main model)
This is the scalability and huge-context contribution. The normative spec is `docs/11-engine-spec.md` (v0.3, normative). Since 2026-09-24 the Track B read-out is the Count-Vote Tsetlin Machine (CVTM) of docs/11 section 11 (docs/12 F19; the six-challenge meanings and the section 11 clarifications are docs/12 F20). Summary:

### Literal space (one-hot, position-bound, no hashing, fully readable)
- Symbols (uint16): token ids [0,V); teacher-free hierarchical cluster ids at 4/8/12-bit prefixes (16/256/4096 ids; in-house 3-level 16-way exchange clustering on left+right bigram contexts, train split only, seeded), which give OR-over-synonyms literals and backoff; specials (MASK, DOC_EDGE, and the polarity symbols from the initial design).
- Segments: 0-15 = exact offsets -8..-1, +1..+8. 16-41 = MULTI-SCALE BAG buckets (8*2^k, 8*2^(k+1)] per side, k = 0..12, reaching N = 65536 tokens, document-bounded, the 64 most frequent tokens removed, tokens + 8-bit cluster symbols.
- `lit = (seg << 17) | (sym << 1) | neg` (uint32). No feature matrix is ever materialized: exact literals are array lookups on the device-resident corpus; bag literals use an occurrence CSR + binary search in training and per-bucket last-occurrence tables in streaming inference (O(log N) maintenance per token, O(1) membership).
- Masking is a pure function: `masked(q) = mix64(seed, epoch, q) < rate_q32`.
- Context-cost rule: every clause MUST have an exact-offset positive anchor; bag and negated literals are only verified, never enumerated. That is what is meant to keep cost per token nearly independent of N (a target, measured in s2). The rule says nothing about clause count: candidates per example still grow with C (s1, KB4). Pure-bag "pointer clauses" (a logical cache LM) are an ablation.

### Clause storage (identical layout on CPU and GPU, about 290 B per clause [derived])
`lit_id [C,32] u32`, `lit_state [C,32] u8` (included iff >= 128), `vote_cls [C,16] u16`, `vote_w [C,16] i16` (SPARSE signed votes instead of a dense V x C matrix), `inc_lit [C,12] u32` + `n_inc [C] u8` (literal budget 12), `anchor [C] u32`, `decay_mark`/`own_fire_mass [C] u64`, `stat_fire/stat_tp/birth_step/last_fire_step [C] u32`, `free_list`, `G [V] u64` (per-class gate mass), `lit_rank`, the anchor inverted index `post_ptr/post_clause` (CSR), `class_bias [V] i32` (log-unigram, calibration only), corpus `tok [N_tok] u16`, `doc_start`, `kind_map [3,V]`, `occ_ptr/occ_pos`, `flags`. Optional wide-vote tier `[C_wide,1024]` for broad distributions.

A printed clause reads: `IF t[-1]=New AND t[+1]=City THEN York(+57), Jersey(+3)`.

### Feedback rules R1..R8 (defaults T=64, s=16, 8 state bits, boost 1; all decisions integer-only)
One line each here; one paragraph each in docs/11.
- R1 events only for (example, clause) pairs where the clause fired, re-verified on live state.
- R2 target class: integer gate; Type Ia and w++ for non-negative or untracked votes, Type II and w++ for negative votes.
- R3 negatives: the 2 top-scoring wrong classes from frozen scores (focused negatives); `uniform_binomial` mode exists for parity tests.
- R4 lazy decay scoped to OWNED classes: this fixes the frequency trap of docs/02.
- R5 seeding of new clauses from a free list, capped per class per batch.
- R6 Type Ia details, including insertion of one sampled untracked active POSITIVE literal.
- R7 Type II details: the ONLY way negated literals enter.
- R8 dense-equivalent mode that must match tmu CoTM, and the DEVIATION LADDER D0..D7 that flips one flag at a time on a tiny LM and logs what each deviation costs or gains (experiment s4).

### Clause sharding
Clauses are independent voters, so the model may be split by clause id range across engine instances or devices. docs/11 section 9 lists exactly what shards exchange (summed sparse scores and cover counts per example, free-list counts for seeding, sweep top-up candidates) and requires the concatenated shard states to hash-equal the unsharded state. On one V100 this is tested with separate instances (s5). It is the mechanism a many-device run would use and the only multi-device statement the project makes.

### Layer 2: the HTM answer (greedy stacking over clause bits)
The engine's clauses are self-supervised concept detectors: layer 1. A second TM reads the fired-clause bit vector of layer 1 (per position for the LM, pooled per paragraph for the proposal task) and is trained with layer 1 frozen: layer 2. This is the engine itself in tabular mode (docs/11 section 6), so layer-2 clauses are readable (`IF clause_17 AND NOT clause_903 THEN same_document`) and carry receipts like any other clause. Joint training of two layers is not attempted: the HGTM experiments of the repository AnwarDebes/HGTM found it collapsed to chance while greedy layer-wise training worked [stated: AnwarDebes/HGTM]. The layer-2 delta is measured, never assumed (experiment H2, pre-registered H4.3).

### Batch semantics
Evaluate a batch against a frozen snapshot, then apply feedback CLAUSE BY CLAUSE, sequentially in example order (only class sums are stale, never a clause's own state), cap 256 events per clause per batch. Batch size B is a measured ablation {1, 64, 256, 1024, 4096} (experiment s3). Batch-synchronous summed TA feedback has no published analysis (docs/01), so this is measured, not assumed.

### Deterministic RNG
Counter-based SplitMix64 finalizer; `rng(seed, stream, a=example uid (epoch<<40|position), b=clause, c=slot)`; stream enum (MASK, GATE_TGT, GATE_NEG, TYPE1_DEC, INSERT, INSERT_PICK, SEED, SEED_PICK, DECAY_ROUND, NEG_DRAW, VETO, SAMPLER); Bernoulli = `u32 < p_q32`; epoch order = 4-round Feistel permutation with cycle walking. Independent of batch composition, thread layout, grid/block. Golden vectors come from the oracle. (Track A keeps curand, seeded before import, with `(seed, grid, block)` logged.)

### Implementations, in build order
1. `oracle.py`: pure Python, sequential, the executable spec (B=1 and batch mode).
2. C++17 + OpenMP host engine, plain C ABI, built by a small `build.py` calling `g++ -O3 -march=native -fopenmp -shared -fPIC`, loaded with `ctypes` (no cmake, no pybind11); debug build with `-fsanitize=address,undefined`.
3. CUDA engine: CuPy 14 for arrays / memory pool / `sort` / `cumsum` / `flatnonzero`; kernels in `kernels.cu` via `cupy.RawModule(backend="nvcc")`; ALL feedback logic in one shared header `tm_core.h` compiled by both g++ and nvcc. No device atomics. Fallback if CuPy misbehaves: the same `.cu` under PyCUDA with host sorting.

GPU kernels: K1 build_windows, K2 count_candidates (+cumsum), K3 expand_verify, K4 compaction, K5 score_accumulate (one thread per example, own row), K6 select_targets_negatives (S_y, gates, top-2 wrong, top-5, seed_req), K7 sort fired keys by (clause, example), K8 segment_bounds, K9 clause_feedback (one thread per touched clause, R1-R7 sequential under the cap), K10 gate mass add, K11 seed_clauses, K12 rebuild_index, K13 sweep, K14 predict_topk / explain.

### Track B data flow
```
raw text -> BPE (8k..16k, train split only) -> tok [N_tok] u16, doc_start     (device-resident corpus, no feature matrix)
  -> K1 build_windows: target positions from the pure masking function (or every position, causal)
  -> K2..K4: anchor inverted index -> candidate clauses -> verify literals on the corpus -> fired (example, clause) pairs
  -> K5: sparse signed votes -> integer class scores;  K6: target gate, top-2 wrong classes, top-5, seed requests
  -> K7..K11: sort by clause, clause-sequential feedback R1..R7, gate mass, seeding;  K12, K13 every 256 batches
  -> the common interface -> Track C (heads, calibration, decoders, receipts, evals)
```

## Three trainings (same engine, same interface)
1. MASKED: mask rate 0.15, comparable to BERT. This is E3.
2. DIFFUSION: mask rate drawn from U(0,1), the initial design's MDLM/RADD model with an ELBO bound.
3. CAUSAL: left segments only, every position a target: a true autoregressive TM LM with a real perplexity.

## Probabilities and perplexity
`logits = a*scores + b*class_bias`, two scalars fitted on validation NLL; split-conformal sets as the ConfTM stand-in; report ECE. Perplexity is reported as bits per byte AND per word (labelled open-vocabulary, not like-for-like with published closed-vocabulary numbers); the in-house KN-5 on identical tokens is the only like-for-like comparator (docs/06).

## Track C: written once against the interface
- Baselines measured in-house on identical tokenization: unigram, left/right bigram, interpolated KN up to order 5 (validated against NLTK on a toy corpus), bidirectional fill-in-the-blank n-gram, in-house CBOW-kNN (the clean stand-in for GloVe), a small transformer MLM trained from scratch on the same tokens capped at 24 GPU-h (the contamination-free neural comparator), BERT-base MLM on whole-word targets labelled "contaminated reference".
- Heads: flat; compositional K=8 x M=32 on cluster codes; additive two-level label tree; the initial design's 3-point kill criterion.
- Calibration: the two-scalar fit above, split-conformal sets, ECE.
- Decoders: masked-diffusion sampler with confidence-ordered unmasking and remasking; any-order AR; elimination decoder with three-valued clause evaluation and PySAT propagation; incremental re-evaluation.
- Certification: see "Certification spec" below.
- Evals: negation minimal pairs and NaN-NLI with sensitivity AND correctness; CondaQA run once and reported honestly; BLiMP by pseudo-log-likelihood; frozen clause-code transfer on R8/IMDB; concept dictionary with WordNet purity and an empirical test of the "216 variants" claim; NVML energy per token under an exclusive GPU lock for TM / small transformer / BERT-base on the same positions in batch-1 and max-throughput regimes; crystallization with absorbing thresholds 50/75/100.
- LEAKAGE RULE: parser-derived features (spaCy negation scope etc.) are an ablation computed from MASKED text only; token-only features come first.

## Component interfaces (signatures; the initial design's list, moved under the `tmlm` package)
```
tmlm.common.interface.Model: predict_scores, logprobs, explain, state_hash, export_bank
tmlm.featurizer.build_windows(corpus, tokenizer, window=24, mask_rate=0.15) -> List[Window]      # Track A
tmlm.featurizer.window_to_graph(window, symbols) -> node/edge/property call sequence           # Track A, Phase 0 scale
tmlm.graphs.build_masked_graphs(windows, hv_size=2048, hv_bits=8) -> Graphs, masked_position   # ONE mask per graph
tmlm.heads.CompositionalHead(K=8, M=32).fit(clause_codes, target_ids); .predict(clause_codes) -> dist over V
tmlm.heads.CoherenceHead.curriculum(positives, tiers) -> (examples, y_coherence)
tmlm.decode.diffusion_sample(model, head, seed_context, steps, temperature) -> token sequence
tmlm.decode.eliminate(bank, partial_assignment) -> licensed_tokens, eliminating_clauses
tmlm.eval.masked_token_topk(model, head, eval_positions, k) -> (top1, top5)
tmlm.eval.energy_per_token(model, corpus) -> joules_per_token   # NVML under the exclusive GPU lock; CodeCarbon as a cross-check
tmlm.certify.receipt(token, context, clauses_fired) -> receipt; certify/checker_standalone.py verifies it
```

## Featurizer spec (now an ablation, from MASKED text only)
BPE vocab 8k to 16k. spaCy neg arcs: any token with `dep_ == "neg"` puts its governor and the governor's subtree content words under negation scope; those tokens use the `neg_tok_i` symbol. Scope adverbs {only, already, still, yet} are bound to their governor as extra node properties. Discourse connectives are their own symbol class. Edges: dependency edge types from `token.dep_`; signed distance edges for offsets -8..-1 and +1..+8 (span 8). This mirrors the GraphTM sequence demo, where the hypervector is doubled so the second half is always the negation of the first half [verified in paper: 2507.14874 "hypervector is doubled in size ... second half is always the negation of the first half"]. Because the parser sees the sentence, these features leak the target unless they are computed from the MASKED text: that is the leakage rule, and it is why token-only features come first and polarity on/off is an ablation (T1.2, M9).

## Compositional-code head spec
Each vocabulary id is assigned K=8 codes, one from each of M=32-way codebooks (K x M scheme), following Shu and Nakayama [stated]. Codebooks are initialised from the teacher-free cluster codes of the literal space (the initial design proposed clustering clause-code embeddings; the Clusters paper shows TMs absorb cluster structure) [stated]. Prediction: the head outputs K softmax-free code logits; decode by nearest-neighbour over the code table (the resonator decoder is deferred, docs/12). Fallback order if accuracy lags: learned codebooks, then the additive two-level label tree. Output-layer memory is K x M x clauses instead of V x clauses, removing vocabulary from the output weights [derived]. In Track B the votes are already sparse, so the head is an ablation against the flat sparse-vote output, not a memory necessity.

## Coherence head and curriculum spec
One extra output column predicts document coherence. Four-tier negative curriculum, all labelled 0: same-topic different document; non-adjacent same document; order-swapped; counterfactually edited positives (from tm-llm-robustness-benchmark). This avoids the NSP-collapses-to-topic failure that made RoBERTa drop NSP and ALBERT switch to SOP [stated]. Tiers 1 to 3 are in scope (M9); tier 4 needs the external repository tm-llm-robustness-benchmark.

## Composition spec (TM Composites, no joint training) [stated; deferred, see docs/12]
- Specialist A: the hierarchical package with OR_ALTERNATIVES at reduced vocab. Since 2026-09-22 it is a MEASURED baseline row on the slide-11 task of the supervisor's project proposal (unpublished; experiment H1, docs/12 F1), not an optional demo.
- Specialist B: GraphTM with typed dependency edges.
- Specialist C: GraphTM with signed distance edges, span 8.
The OR port into GraphTM layer zero is deferred: the initial design's kill-criterion path is taken from the start. Challenge C1 is delivered by cluster literals and vote-set OR groups in Track B.

## Diffusion sampler spec
Loss: MDLM weighted masked cross-entropy (the ELBO reduces to a weighted sum of per-position masked cross-entropies) [verified in paper: MDLM]. Sampling: start fully masked, unmask positions in order of clause-vote margin (highest confidence first), remask low-confidence positions, iterate. Vote calibration: the two-scalar fit above (minimal) or the split-conformal wrapper (defensible). Use incremental re-evaluation per step because consecutive partial inputs are similar [stated]. Prompts are PRE-REGISTERED (first K test documents, seeds 0..k) and every sample is logged.

## Certification spec (receipt format)
A receipt = the context assignment + the fired clauses with their literals and votes + the claimed token. `checker_standalone.py` imports nothing from the project and verifies that (1) each listed clause is satisfied by the context, (2) no unlisted clause of the exported bank fires, and (3) with z3 integer arithmetic, the claimed token is the strict argmax of the votes. Tampered receipts must fail. For the elimination decoder, record the positive-polarity clauses that fired (licensing) and the negative-polarity clauses that fired (eliminating); the bank is encoded as a CNF for PySAT propagation. This is checkable independently of the TM. d-DNNF marginals are deferred (optional, toy bank only).

## Six-challenge mapping on one shared clause pool [stated; superseded by docs/12 F20: Granmo's own meanings are measured by docs/13 A4, H10]
C1 Conceptualization: OR alternation through cluster literals and vote-set OR groups (Specialist A optional). C2 Introspection: vote margin as competence, split-conformal sets, drives unmasking order. C3 Transparency: printed clauses plus receipts. C4 Lifelong learning: crystallization with absorbing thresholds, clause recycling in the sweep. C5 Generalization: cluster-symbol backoff literals; competence-weighted composition is deferred. C6 Reactive and reflective: single denoise pass vs iterative remask loop.

## Memory and compute budgets (32 GB)
Track A: model under 2 GB at HV 2048 / 8192 clauses / 16k outputs, counting the second dense update buffer [derived, docs/02]. Class-loop cost about 30 min per 1M-example epoch at 16k outputs before the Binomial draw; the draw cuts the touched class set to about 6400 at Q=1 (2.5x) or about 800 at Q=0.1 (20x) [derived]. Throughput: stock 20 to 150 examples/s at V=16k, patched 1k to 3k examples/s [derived, unverified].

Track B, ALL [derived, unverified], replaced by measurements at M3/M4:
| Clauses C | Bank memory | V100 training | 6-CPU inference |
|---|---|---|---|
| 8k | 2.4 MB | | |
| 1M | 0.29 GB | 50k to 150k examples/s | 40k to 80k tokens/s |
| 16M | 4.6 GB | 15k to 30k examples/s | |
| 64M | 18.4 GB (still fits 32 GB) | | |

One epoch = every position, about 130M BPE examples; at C=1M about 15 to 45 min on the GPU [derived, unverified]. Host memory is capped at 85 GiB by `mem_guard` (the cgroup limit is 96 GiB of RAM with no swap).
