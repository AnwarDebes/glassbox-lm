# 02 Verified code facts and APIs

Verification status: the initial design was written without access to `tm.py`, `kernels.py`, or `autoencoder.py` and labelled their contents [stated]. They have now been read from source. GraphTM facts below come from the vendored, unmodified master at `third_party/GraphTsetlinMachine` (commit c6129e14, MIT licence) and carry `[verified in source: <file>:<line>]`, with paths relative to `third_party/GraphTsetlinMachine/GraphTsetlinMachine/`. tmu facts come from the installed tmu 0.8.3, hierarchical-package facts from the installed PyHierarchicalTsetlinMachineCUDA 0.3.1 (git c85ca4e7) [measured: `results/_bootstrap/tm_env_install.log`]. Protocol rule 10 still applies: milestone M0 prints `inspect.signature(...)` for every class used and records the printout under `results/*_envcheck/`. Where the initial design's text was wrong it is corrected here and the change is listed in docs/12.

## V1. GraphTsetlinMachine (PyPI 0.3.3; master 0.3.4, vendored)

### Version facts [verified in source]
PyPI latest is 0.3.3, released Apr 4 2025; GitHub Releases lists 0.3.3 as latest; the README install line names `GraphTsetlinMachine-0.3.4.tar.gz` as an sdist build target, so 0.3.4 is master-only [verified in source: PyPI page, README]. Roadmap lists "Rewrite graphs.py in C or numba", "Add autoencoder", "Add regression", "Add multi-output", "Graph initialization with adjacency matrix" [verified in source: README roadmap]. The roadmap trails the code: `MultiOutputGraphTsetlinMachine` EXISTS in master [verified in source: tm.py:961] and in 0.3.3 [stated: design notes, source diff]. The training and inference kernels are byte-identical between 0.3.3 and master; master adds a `get_hyperliterals` kernel, `Graphs.__getitem__`, and save/load fixes [stated: design notes, source diff]. We vendor master under `third_party/` and install it from there (it reports version 0.3.4) [measured: `results/_bootstrap/tm_env_install.log`]. The install is EDITABLE [verified in source: `.venv-tm/lib/python3.11/site-packages/graphtsetlinmachine-0.3.4.dist-info/direct_url.json` holds `"editable":true`], so `import GraphTsetlinMachine` resolves into `third_party/GraphTsetlinMachine/` itself: never edit that tree, or the "stock" reference changes silently.

### Graphs class build order [verified in source: README tutorial]
Constructor:
```
graphs_train = Graphs(
    number_of_graphs,           # e.g. 10000
    symbols = ['A', 'B'],       # list of all symbols
    hypervector_size = 32,      # bits in the hypervector (source default 128)
    hypervector_bits = 2        # bits set per symbol; gives C(size,bits) unique codes
)
```
The README states the codes are "sparse distributed codes" generated and compiled automatically at initialisation, giving size*(size-1)/2 unique bit pairs at hypervector_bits=2 (496 for size 32) [verified in source]. The keyword arguments `double_hashing=False`, `one_hot_encoding=False`, and `init_with=None` exist [verified in source: graphs.py:30-35].

Exact call order [verified in source: README tutorial]:
```
for graph_id in range(number_of_graphs):
    graphs.set_number_of_graph_nodes(graph_id, n_nodes)
graphs.prepare_node_configuration()
for graph_id in range(number_of_graphs):
    graphs.add_graph_node(graph_id, node_name, number_of_outgoing_edges)
graphs.prepare_edge_configuration()
for graph_id in range(number_of_graphs):
    graphs.add_graph_node_edge(graph_id, source_node_name, destination_node_name, edge_type_name)
    # add reverse edge for the other direction; edges are directed
graphs.add_graph_node_property(graph_id, node_name, symbol)   # may be called multiple times per node
graphs.encode()
```
For a test set, construct with `Graphs(n_test, init_with=graphs_train)` so the same hypervector codebook is reused [verified in source: graphs.py:35].

Edge types are bound to messages by vector-symbolic binding: a matched clause C sends message M bound to the edge type t as M (x) t (the (x) operator is binding), so a receiving node sees a feature named `edge_type (x) clause` [verified in source: README "Left (x) C" and "Right (x) C" features, and arXiv 2507.14874 step 2]. In the kernel the binding is a cyclic shift of the message's sparse hypervector bit positions by the edge-type id: `shifted_bit = (bit + edge_type) % MESSAGE_SIZE` [verified in source: kernels.py:581].

### MultiClassGraphTsetlinMachine and MultiOutputGraphTsetlinMachine (tm.py)
`MultiClassGraphTsetlinMachine` [verified in source: tm.py:903] and `MultiOutputGraphTsetlinMachine` [verified in source: tm.py:961] both exist and accept `grid` and `block` as 3-tuples. README CUDA configurations: `grid=(16*13,1,1), block=(128,1,1)` for A100, `grid=(16*13*4,1,1)` for H100 [verified in source: README CUDA Configurations].

Constructor, identical for all classes [verified in source: tm.py:38-55, tm.py:908-924, tm.py:966-982]:
```
MultiClassGraphTsetlinMachine(          # same for MultiOutputGraphTsetlinMachine, GraphTsetlinMachine
    number_of_clauses, T, s,
    q=1.0,
    max_included_literals=None,
    boost_true_positive_feedback=1,
    number_of_state_bits=8,
    depth=1,
    message_size=256,
    message_bits=2,
    double_hashing=False,
    one_hot_encoding=False,
    grid=(16*13*4,1,1),                 # H100 default: ALWAYS override on the V100, see blocker 4
    block=(128,1,1)
)
```
There is NO `append_negated` keyword argument: the initial design's constructor was wrong. Negated literals are always present (the hypervector is doubled).

Methods:
```
fit(graphs, Y, epochs=100, incremental=False)   # [verified in source: tm.py:943, tm.py:1001]; the initial design's epochs=1 default was wrong
predict(graphs)
score(graphs)                                   # class sums
transform(graphs)                               # returns a TUPLE (clause_outputs, class_sum) [verified in source: tm.py:814-851]
transform_nodewise(graphs)                      # per-node clause outputs [verified in source: tm.py:853]
```
Y encoding: `MultiClassGraphTsetlinMachine.fit` takes class ids of shape (n_examples,) and builds a dense `encoded_Y` of shape (n_examples, n_outputs), int32, +T for the true class and -T elsewhere [verified in source: tm.py:949-951]. It sets `number_of_outputs = max(Y) + 1` [verified in source: tm.py:944], so CLASSES silently shrinks if the largest vocabulary id never occurs in Y: check this in every harness. `MultiOutputGraphTsetlinMachine.fit` takes Y of shape (n_examples, n_outputs), binary 0/1, encoded to +/-T per output [verified in source: tm.py:1007]. `_init_gpu_kernels` injects `#define CLASSES %d` at compile time with no hard cap [verified in source: tm.py:437-450]. The `select_clause_updates` gate is [verified in source: kernels.py:384]:
```
if ((target == -1 && curand_uniform(&localState) > 1.0*Q/max(1, CLASSES-1)) ||
    (curand_uniform(&localState) > 1.0*absolute_prediction_error/(2*THRESHOLD)))
    class_clause_update[class_id*CLAUSES+clause] = 0;   // Q default 1.0
```
It fires per (clause, class) pair [verified in source: kernels.py:371-401]. Read it carefully, because the initial design describes it wrongly. For a negative class (`target == -1`) an update needs BOTH gates to pass: the Q gate (first uniform <= Q/(CLASSES-1)) AND the error gate (second uniform <= err/(2T), with err = |y - clamp(class_sum)| = T + clamp(class_sum) for a negative class). The `||` short-circuits, so the second uniform is drawn only for pairs that passed the Q gate. The probability of a negative update is therefore Q/(CLASSES-1) x err/(2T), not Q/(CLASSES-1) [derived from kernels.py:384]. A negative class whose sum is clamped at -T has `target == +1` and err = 0 and is never updated [verified in source: kernels.py:380-384]. `curand_uniform` returns values in (0, 1] [verified in source: `/usr/local/cuda/include/curand_uniform.h:123`, "Output range excludes 0.0f but includes 1.0f"], so an error-gate threshold of 0 always blocks and a threshold of 1 never does. The true class skips the Q gate and is updated with probability (T - clamp(class_sum))/(2T).

Printing clauses: there is NO clause printer in the package. Use `ta_action(depth, clause, ta)` [verified in source: tm.py:126], `get_hyperliterals(depth)` [tm.py:144], `get_clause_literals(graphs.hypervectors)` [tm.py:213], and `get_weights()` [tm.py:207]; the demos print clauses by looping over `ta_action` [verified in source: examples/NoisyXOR_print_clauses.py and others]. `get_state()` copies only `ta_state` and `clause_weights` and so drops the message-layer TAs [verified in source: tm.py:281-287]: persist models with `save()` / `load()` [tm.py:312, tm.py:364].

### Minimal end-to-end examples

(a) Build a token graph with dependency and distance edges and polarity symbols (ONE masked position per graph, see "Graph-level, not node-level, classification" below):
```
V = 8000
symbols = [f"tok_{i}" for i in range(V)] + [f"neg_tok_{i}" for i in range(V)] \
          + ["only","already","still","yet"] + ["MASK"] + [f"disc_{c}" for c in connectives]
graphs = Graphs(n_windows, symbols=symbols, hypervector_size=2048, hypervector_bits=8)
for gid, window in enumerate(windows):
    graphs.set_number_of_graph_nodes(gid, len(window))
graphs.prepare_node_configuration()
for gid, window in enumerate(windows):
    for pos, tok in enumerate(window):
        graphs.add_graph_node(gid, f"n{pos}", out_degree(pos, window))
graphs.prepare_edge_configuration()
for gid, window in enumerate(windows):
    for (src, dst, dep_label) in dependency_edges(window):      # spaCy dep_ as edge type (ablation only, from MASKED text)
        graphs.add_graph_node_edge(gid, f"n{src}", f"n{dst}", f"dep_{dep_label}")
    for pos in range(len(window)):                              # signed distance edges, span 8 (banded, not fully connected)
        for d in range(1, 9):
            if pos+d < len(window):
                graphs.add_graph_node_edge(gid, f"n{pos}", f"n{pos+d}", f"dist_+{d}")
                graphs.add_graph_node_edge(gid, f"n{pos+d}", f"n{pos}", f"dist_-{d}")
    for pos, tok in enumerate(window):
        sym = "MASK" if pos == masked_position[gid] else (f"neg_tok_{tok}" if under_neg_scope(pos) else f"tok_{tok}")
        graphs.add_graph_node_property(gid, f"n{pos}", sym)
        for adv in scope_adverbs_bound_to(pos): graphs.add_graph_node_property(gid, f"n{pos}", adv)
graphs.encode()
```

(b) Train with ONE target token per graph. At small scale the stock multi-class model takes class ids directly:
```
import numpy as np
np.random.seed(seed)                       # BEFORE importing GraphTsetlinMachine (blocker 5)
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
# y[i] = vocabulary id of the true token at the masked node of window i
tm = MultiClassGraphTsetlinMachine(number_of_clauses=8192, T=..., s=...,
        depth=3, message_size=1024, message_bits=..., q=1.0,
        max_included_literals=..., grid=(16*13,1,1), block=(128,1,1), number_of_state_bits=8)
tm.fit(graphs, y, epochs=1, incremental=True)
```
The stock path builds a dense `encoded_Y` of N x CLASSES int32 (blocker 1), so at corpus scale only the patched fork with a sparse `y_target[N]` (docs/03, Track A) can run this.

(c) Extract clause-codes: `codes, class_sums = tm.transform(graphs)`  # `codes` is the per-example clause activation matrix (n_graphs x n_clauses, int32), used as a frozen embedding for R8/IMDB transfer.

(d) Print clauses: for each clause and each literal index k, test `tm.ta_action(depth, clause, k)`; map included hypervector bits back to symbols with `get_clause_literals(graphs.hypervectors)`; read votes from `get_weights()`. No per-clause printer is provided by the package.

### Kernel edit points for the Binomial draw (kernels.py) [verified in source]
`CLASSES` is in the innermost loop of four kernels: `evaluate` (kernels.py:311, with `atomicAdd` to class_sum per active clause per class at kernels.py:313), `select_clause_updates` (kernels.py:372, the gate above at kernels.py:384), `update` (kernels.py:236; `update_clause` falls through on zero target_sign), and `update_message` (kernels.py:205). The initial design listed three sites; `update_message` is the fourth. `_fit` processes one graph at a time [verified in source: tm.py:664].

Reformulation (exact, lossless): the expected number of Q-gate selections per example (negative (clause, class) pairs that pass the Q gate) equals CLAUSES x Q. The initial design called this "expected negative updates", which is wrong: a selection becomes an update only if it also passes the error gate, so realised negative updates per example have mean CLAUSES x Q x mean(err/(2T)), about CLAUSES x Q / 2 when class sums are near 0 (err = T) [derived from kernels.py:384]. Drawing k ~ Binomial(CLASSES-1, Q/(CLASSES-1)) negative classes per clause, touching only those, and applying the unchanged error gate to each drawn class is distributionally identical to the dense gate. This removes the CLASSES factor from `select_clause_updates`, `update`, and `update_message`. For `evaluate`, compute class_sum lazily only for the true class plus the union of drawn classes. At Q=1, 8192 clauses, 16000 classes the union is about 6400 (2.5x fewer than dense); at Q=0.1 about 800 (20x) [derived, task B2]. Sweep Q.

Exact Binomial, not Poisson(Q): the initial design wrote "Binomial ~ Poisson(Q)". At CLASSES=100 the Poisson approximation differs by about 1% in variance, which the initial design's own 200k-trial chi-square can flag [derived: design notes]. Concretely, at CLASSES=100, Q=1, M=200000, a CORRECT exact-Binomial sampler tested against a Poisson(1) reference is rejected at the 0.01 level with probability about 0.39 instead of 0.01 [derived: noncentral chi-square, noncentrality M x sum((b_k - p_k)^2 / p_k) = 10.2 over 8 bins pooled to an expected count of at least 5, 7 degrees of freedom]. At CLASSES=1000 and 16000 the two references are indistinguishable at this M (noncentrality at most 0.1) [derived]. The patched kernel draws the exact Binomial by inversion against a host-computed Q64 CDF table, with K_MAX=32 distinct negatives per clause (the tail mass beyond 32 is below 1e-35 at Q <= 1 [derived]).

Before/after pseudocode:
```
# BEFORE (select_clause_updates), per clause, loops all CLASSES-1 negatives:
for class_id in 0..CLASSES-1:
    if class_id == true_class: continue
    if (curand_uniform() > Q/(CLASSES-1)) or (curand_uniform() > err/(2T)):
        class_clause_update[class_id*CLAUSES+clause] = 0

# AFTER, per clause, draw k ~ Binomial(CLASSES-1, Q/(CLASSES-1)) exactly, sample k distinct negatives, touch only those:
k = binomial_by_inversion(rng, cdf_table_q64)        # k <= K_MAX = 32
for j in 0..k-1:
    class_id = sample_distinct_negative(rng, true_class, CLASSES)
    if (curand_uniform() > err/(2T)):            # the error gate is UNCHANGED: a drawn class is a Q-gate selection, not yet an update
        skip
    else:
        mark class_id as active-for-update; set class_needed[class_id]
# evaluate_sparse: accumulate class_sum only for {true_class} union {drawn class_ids}, one thread per needed class, no atomics
# the dense path stays available behind NEG_DRAW=dense
```

### Distributional-equivalence test design (tests/test_binomial_draw.py)
Run both the dense gate and the sparse draw for M=200000 seeded trials at Q in {0.1,0.5,1.0}, CLASSES in {100,1000,16000} (the initial design's grid). The initial design's assertions (1) and (2) are kept, with two corrections (docs/12, B15 and C2).

What is counted. Assertions (1) and (2) are about Q-gate SELECTIONS (drawn negatives), not about realised updates. The stock kernel exposes only the result of both gates (`class_clause_update`), so the harness forces the error gate open: every negative class sum is set to +T, which gives err = 2T and err/(2T) = 1, and `curand_uniform` never exceeds 1. Under that setting every selection is an update, and the nonzero negative entries of `class_clause_update` per clause are exactly the selections. On the sparse path the same count is `neg_cnt`.

Assert:
1. Mean Q-gate selections per clause equals Q within 3 standard errors (error gate forced open).
2. The per-clause selection-count histogram matches Binomial(CLASSES-1, Q/(CLASSES-1)) by a chi-square test at p>0.01 (error gate forced open). The initial design named Poisson(Q) as the reference; the exact Binomial is the correct one, and a Poisson reference would reject a correct sampler (see above).
3. With a fixed seed, the set of touched (clause, class) pairs is a valid subset and the resulting `ta_state` after one step is statistically indistinguishable (two-sample test on state deltas) from the dense path aggregated over seeds.
4. (Added.) The error gate is preserved: with every class sum at 0 (err = T, err/(2T) = 1/2), realised negative updates per clause have mean Q/2 within 3 standard errors on BOTH paths, and their count matches Binomial(CLASSES-1, Q/(2(CLASSES-1))) (an independent thinning of a Binomial is a Binomial) [derived].

Taken literally, the initial design's assertion (1), "mean negative updates per clause equals Q", FAILS for a correct kernel at any class sums other than +T: realised updates have mean Q x mean(err/(2T)). If this test and the stock kernel ever disagree, the stock kernel (kernels.py:384) is the reference and the test is what gets fixed. Never change a kernel to make the test pass (protocol rule 7). The test covers the dense and the sparse paths.

### Memory formulas [derived; formulas confirmed against source, design notes]
```
ta_chunks           = ceil(2 * hypervector_size / 32)          # factor 2: negated literals are always appended
ta_state_bytes      = clauses * ta_chunks * number_of_state_bits * 4
clause_weights      = outputs * clauses * 4
class_clause_update = outputs * clauses * 4                    # a SECOND dense buffer [verified in source: tm.py:555]
encoded_Y           = examples * outputs * 4                   # dense targets, host and device [verified in source: tm.py:557-561]
message_chunks      = ceil(2 * message_size / 32)
message_ta          = layers * clauses * message_chunks * number_of_state_bits * 4
```
At HV 2048, 8192 clauses, 8 state bits: ta_chunks=128, ta_state = 8192*128*8*4 = 33.5 MB; clause_weights at 16000 outputs = 16000*8192*4 = 524 MB, and class_clause_update is another 524 MB; the model stays under 2 GB [derived]. `encoded_Y` at 1M examples x 16000 outputs is 64 GB and does not fit [derived]: see blocker 1. Class loop streams about 1.6 GB per example at 16k outputs, about 1.8 ms on 900 GB/s, about 30 min per 1M-example epoch [derived, task B2]; clause evaluation is small.

### Blockers in the stock package (not known to the initial design)
1. Dense `encoded_Y` of shape (N, CLASSES), int32, on host and device [verified in source: tm.py:949, tm.py:1007, tm.py:557-561]. 1M examples x 16k classes = 64 GB [derived]. There is no sparse Y.
2. `_fit` AND `_score` run one graph at a time from Python [verified in source: tm.py:664, tm.py:786]: 10 (depth 2) to 14 (depth 3) kernel launches per example [stated: design notes, count], each followed by `cuda.Context.synchronize()` [verified in source: tm.py:701-747]. There is no shuffling [verified in source: no shuffle or permutation call in tm.py].
3. `Graphs` construction is pure Python, about 2k to 4k graphs/s [unverified, inferred; measure in M1]. Clean bypass: fill `X`, `edge`, `node_type`, `number_of_graph_node_edges`, `node_index`, `edge_index` directly and set `signature` by hand (the model re-uploads only when the signature changes) [verified in source: graphs.py:223, tm.py:528-529].
4. The default grid `(16*13*4,1,1)` is the H100 setting [verified in source: tm.py:53] and likely indexes past the curandState array on a V100 [unverified]: ALWAYS pass `grid=(16*13,1,1)`.
5. curand is seeded from global `np.random` at IMPORT time, by the module-level `g = curandom.XORWOWRandomNumberGenerator()` [verified in source: tm.py:36]: seed `np.random` before importing. Results depend on `(seed, grid, block)`; log all three.
6. The whole dataset `X` is uploaded to the GPU at once [verified in source: tm.py:531-532]: 12 KiB per 24-node graph at HV 2048 [derived], so shard the corpus.
7. Stock `evaluate` calls `atomicAdd(int*, int)` [verified in source: kernels.py:313]. Whether the stock kernels compile under nvcc 12.9 is [unverified]: it is the first Phase 0 action (M0 microtest, M1 T0.3).

### Graph-level, not node-level, classification
GraphTM classifies GRAPHS, not nodes: a clause's output is the OR over all nodes (`evaluate` adds the clause weight once if any node chunk is set [verified in source: kernels.py:300-313]) and Y has one row per graph. The initial design's "multiple masks per input" (T1.3) would therefore train an unordered multi-hot target: the model could not tell which masked node a predicted token belongs to. Track A uses ONE target per graph.

### The frequency trap (coalesced TM feedback at language-model sparsity)
Every clause with non-negative weight for the target class gets Type I feedback from every such example, and a non-firing clause decays all its literals at rate 1/s [verified in source: kernels.py:115, kernels.py:159]. An included literal therefore survives only if the clause's firing rate exceeds about 1/(s+1) [derived]. Language-model patterns fire at rates of 1e-5 to 1e-3, so at s = 5..10 only patterns at the level of "t[-1]=the" survive [derived]. Consequences: E1 must sweep s in {10, 30, 100, 300, 1000, 3000} together with `max_included_literals`; Track B fixes the trap structurally with lazy decay scoped to owned classes (docs/11, R4).

### Other semantics worth knowing
- A clause fires at depth d only where it fired at depth d-1: the deeper layers AND their output with the previous layer's node output [verified in source: kernels.py:470, kernels.py:525-527].
- GraphTM's own NLP setup in the paper is a FULLY CONNECTED word graph, O(n^2) edges [stated: design notes]. Track A uses one shared BANDED signed-distance edge template instead.
- No IMDB or other text demo exists in the repository [verified in source: `third_party/GraphTsetlinMachine/examples/` holds XOR, OR, MNIST, CIFAR-10, and synthetic sequence demos only].

## V2. tmu TMAutoEncoder [verified in source: tmu 0.8.3, `tmu/models/autoencoder/autoencoder.py`; verified in paper: arXiv 2301.00709]

tmu version in use since 2026-09-22: upstream commit 5605ff070a18549328028c907a9acf68e063346e (2026-08-03, still reports 0.8.3), built from source, replacing the 0.8.3 wheel of 2024-03-25. Reason: the wheel's `ClauseBank.initialize_clauses` executes `np.uint32(~0)`, which NumPy 2 rejects (`OverflowError: Python integer -1 out of bounds for uint32`), so no tmu classifier can be constructed under the installed numpy 2.4.6 [measured: the failing test run is quoted in docs/results.md]; the M1 autoencoder reproduction had worked around it with a runtime source patch (`experiments/p0_tmu_ae_repro.py`, `apply_numpy2_shim`), which is now unnecessary. Upstream fixed it in commit 62c0e6a6 (2024-11-22). Pitfall [measured, 2026-09-22, exploratory run]: passing `seed=` to `TMCoalescedClassifier` makes its CPU training about 50 times slower (one epoch of 2000 examples went from 0.2 s to over 10 minutes), so the parity protocol runs tmu unseeded. The diff v0.8.3 to 5605ff07 over the files we use is 17 insertions and 7 deletions: the NumPy 2 fix in `clause_bank.py`, a mask fix in `base_clause_bank.py`, two lines in `autoencoder.py`, three in `models/base.py` [verified in source: `git diff v0.8.3 HEAD` in `third_party/ref/tmu`]. Install log: `results/20260922T145836Z_tmu_pin_change/`.
Architecture: Coalesced TM extended with input masking and freezing of masked variables; input and output are propositional bag-of-words of length 2V (word presence and its negation); objective is document-level presence of one target word, not MLM; static embeddings [verified in paper: 2301.00709 Section 3]. Fit loop [verified in source: autoencoder.py:248-277]:
```
for e in range(number_of_examples):
    for i in class_index:                       # each target word
        Xu, Yu = self.clause_bank.produce_autoencoder_example(target=i, ..., accumulation=self.accumulation)
        self.update(i, Yu, Xu, ...)
```
The whole clause bank is re-evaluated V times per round; cost is linear in vocabulary. The initial design's premise holds. Constructor [verified in source: autoencoder.py:25-48]:
```
TMAutoEncoder(number_of_clauses, T, s, output_active, accumulation=1,
    type_i_ii_ratio=1.0, type_iii_feedback=False, focused_negative_sampling=False,
    output_balancing=0, upsampling=1, d=200.0, platform='CPU', patch_dim=None,
    feature_negation=True, boost_true_positive_feedback=1, reuse_random_feedback=0,
    max_included_literals=None, number_of_state_bits_ta=8, number_of_state_bits_ind=8,
    weighted_clauses=False, clause_drop_p=0.0, literal_drop_p=0.0, ...)
fit(X, number_of_examples=2000, shuffle=True)
```
Corrections to the initial design: the `accumulation` DEFAULT is 1, not 25 (25 is the value the TM-AE paper and the IMDb demo pass [verified in paper: accumulation 25]); `weighted_clauses` defaults to False, not True. `output_active` is the array of target-word output indices; `accumulation` is the number of documents combined per training example.

`platform='CUDA'` is BROKEN on tmu main: PRs #81 (2025-04-25), #85 and #86 (2026-07-01) are still OPEN on 2026-09-22; the last merged code PR is #79 "Fixing the autoencoder model" (2025-03-04) [verified in source: GitHub, log `results/20260922T125436Z_prior_art_check/A_tm_language_model_prior_art.md`]. The CPU path uses unseeded C `rand()` [stated: design notes], so T0.5 runs on CPU and is recorded as non-reproducible. IMDbAutoEncoderDemo settings: clauses 80, T 160, s 5.0, accumulation 25, 2500 examples, max included literals 3 [stated: design notes, read from the demo]. Inspection methods on the model: `get_weight`, `get_weights`, `literal_importance`, `clause_precision`, `clause_recall` [verified in source: autoencoder.py:312-410]; there is no clause printer.

tmu ships NO R8/R52 loader [verified in source: tmu 0.8.3 `tmu/data/` holds bot_iot, cic_ids, cifar10, cifar100, fashion_mnist, imdb_keras, kdd_99, mnist, nsl_kdd, unsw_nb15 only]. Use the Hugging Face ids in V5.

## V3. PyHierarchicalTsetlinMachineCUDA 0.3.1 [verified in source: installed from git c85ca4e7, which is ahead of PyPI]
Constants: AND_GROUP=0, AND_ALTERNATIVES=1, OR_GROUP=2, OR_ALTERNATIVES=3; VANILLA_TM=0, WEIGHTED_TM=1, COALESCED_TM=2 [verified in source: PyHierarchicalTsetlinMachineCUDA/tm.py:36-43]. Classes: CommonTsetlinMachine, MultiOutputTsetlinMachine, MultiClassCoalescedTsetlinMachine, MultiClassTsetlinMachine, TsetlinMachine, RegressionTsetlinMachine [verified in source: tm.py:45, 660, 686, 718, 807, 840]. This package, unlike GraphTM, does take `append_negated=True` and `seed=None`, and its default grid is `(16*13,1,1)` [verified in source: tm.py:47]. NoisyParityDemo [verified in source]:
```
from PyHierarchicalTsetlinMachineCUDA.tm import TsetlinMachine
import PyHierarchicalTsetlinMachineCUDA.tm as tm
clauses = 32; s = 25.0; T = 250
tm_model = TsetlinMachine(clauses, T, s, number_of_state_bits=8, boost_true_positive_feedback=0,
    hierarchy_structure=((tm.AND_GROUP, 3), (tm.OR_ALTERNATIVES, 3), (tm.AND_GROUP, 2),
                         (tm.OR_ALTERNATIVES, 3), (tm.AND_GROUP, 2)))
for e in range(1000):
    tm_model.fit(X_train, Y_train, epochs=10, incremental=True)
    tm_model.print_hierarchy()
```
HexDemo uses clauses default 2000, T default 9000, s default 44.0, q default 1.0, boost 1, number_of_state_bits default 7, and 60 or_alternatives per the argparse block [verified in source: repo README shows argparse defaults and HexDemo output rising past 99% training accuracy across epochs]. The exact HexDemo final testing accuracy could not be confirmed verbatim; do not quote a specific Hex accuracy until read from a run log [unverified].

What the hierarchy computes [verified in source: `third_party/ref/HierarchicalTsetlinMachine/TsetlinMachine.c` and `.h`, and the 2026-02-06 presentation]: a clause is a tree of clause components (leaves over feature groups of `LEAF_FACTORS` features); at inference "Replace OR with Addition, Replace AND with Multiplication": `leaf_vote_sum` sums the outputs of the alternatives of a group, `interior_vote_products` multiplies the sums over the AND groups, and so on up to `clause_output`, so a clause returns an integer vote count, not a bit. Learning (`tm_update`): every component of every clause draws its own feedback decision with probability `(T + (1 - 2*target) * class_sum) / (2T)`; a component then gets Type I (if its path of AND sub-expressions is true: Type Ia, else Ib) or Type II (only if the path is true). The whole hierarchy is trained jointly per example, not layer by layer. The CUDA package at HEAD 5b4a223 adds `and_group_normalization` (geometric mean instead of a plain product over the AND factors), `constant_update_p` (update probability 1), `no_clipping` (class sums not clipped to +-T) and `binary_inference` [verified in source: `third_party/ref/PyHierarchicalTsetlinMachineCUDA/PyHierarchicalTsetlinMachineCUDA/kernels.py`, `tm.py`]. Experiment H1 sweeps those switches on validation and pins two commits (configs/H1.json).

The hierarchy structure that the supervisor's project proposal (unpublished) gives for its slide-11 task is not reproduced in this document; experiment H1 of [docs/05](05-experiments.md) describes the task and the measurement of the hierarchical package on it.

Repository status on 2026-09-22 [verified in source: GitHub API and raw `tm.py` at HEAD 5b4a223; log `results/20260922T125436Z_prior_art_check/A_tm_language_model_prior_art.md`]: very active (about 1,453 commits, 15 in the two days before the check), version bumped to 0.3.2, live branches `and_vote_normalization*` and `local_update_p*` that change the update rule, README citing "A Tsetlin Machine for Logical Learning and Reasoning With AND-OR Hierarchies", Granmo et al., 2026, forthcoming. The vendored install c85ca4e7 is far behind main. Experiment H1 pins ONE commit, prints its signatures into `results/`, and re-reads `tm.py` before any run. `MultiOutputTsetlinMachine` has THREE defects, not one [verified in source: tm.py:660-663, tm.py:47-55; still present at HEAD 5b4a223 on 2026-09-22]: (1) its `__init__` does not accept or pass `hierarchy_structure`; the parent defaults it to None and runs `self.depth = len(hierarchy_structure)`, so a TypeError (len(None)) is raised; (2) it never sets `flip_polarity`; (3) it never sets `tm_type`. T0.7 documents all three as a failing test. Input width `number_of_features_hierarchy` is the product of AND_GROUP and OR_GROUP factors only, equal to 2V for the slide-11 structure of the supervisor's project proposal (unpublished), i.e. strictly bag-of-words with two slots and no order; `number_of_ta_teams = 2 x alternatives x concepts` [stated, task B1]. TA memory = clauses x ta_teams x ceil(literals_per_leaf/32) x state_bits x 4 bytes [verified in source: tm.py:86-90, tm.py:218; the initial design printed a divisor of 96, which does not reproduce its own table].

Derived memory table (append_negated=False, 8 state bits), rows that fit the 32 GB V100 only [derived]:
| V | alternatives | concepts | clauses | memory |
|---|---|---|---|---|
| 10000 | 10 | 5 | 1000 | 1.0 GB |
| 10000 | 20 | 10 | 2000 | 8 GB |
| 10000 | 60 | 10 | 2000 | 24 GB (little headroom) |
| 30000 | 10 | 5 | 2000 | 6 GB |
| 30000 | 20 | 10 | 2000 | 24 GB (little headroom) |

32 GB envelope for the hierarchical package: vocab 8k to 15k, alternatives 10 to 20, concepts 5 to 10, clauses 1000 to 2000; the largest corner of that box is 12.0 GB [derived]. The wall is alternatives x concepts, not vocabulary. The initial design's V = 30000, alternatives 60 row needs 72 GB [derived] and is out of reach on this machine.

## V4. Installation
There is no root and no system Python or pip, so the initial design's `python -m venv` and `pip install` lines cannot run. The approach: `uv` in `~/.local/bin`, a uv-managed CPython 3.11 (it ships `Python.h`), and two environments. The command block is in the research protocol ("Environment setup"). That block is a RECONSTRUCTION [unverified]: the install commands were never logged, only uv's output was, and `env/bootstrap.sh` (design decision) has NOT been written yet. It has not been re-run from a clean checkout. What IS measured is the set of installed versions:
- `.venv-tm` [measured: `results/_bootstrap/tm_env_install.log`, `results/_bootstrap/pycuda_build.log`]: Python 3.11.16, pycuda 2025.1.3 (built from source), tmu 0.8.3, cupy-cuda12x 14.2.0, numba 0.67.0, z3-solver 5.1.0.0, python-sat 1.8.dev17 (a dev build; the reconstructed block pins it), codecarbon 3.3.1, nvidia-ml-py 13.610.43, datasets 5.0.1, tokenizers 0.23.2, nltk 3.10.3, GraphTsetlinMachine 0.3.4 (editable, from `third_party/`), PyHierarchicalTsetlinMachineCUDA 0.3.1 from git c85ca4e70af90511d6b4f8e8eedccd23a7d7d0ec (the block pins this commit, because V3 cites its line numbers).
- `.venv-tm`, from install steps that were not logged [verified in source: the dist-info directories under `.venv-tm/lib/python3.11/site-packages/`; `make env-check` will log them]: numpy 2.4.6, setuptools 84.0.0, wheel 0.48.0, scipy 1.17.1, pytools 2026.1.1, mako 1.4.1, pytest 9.1.1. numpy, setuptools, and wheel were in place before the pycuda build, which `--no-build-isolation` requires.
- `.venv-dl` [measured: `results/_bootstrap/dl_env_install.log`]: torch 2.14.0+cu126, triton 3.8.0, transformers 5.17.0, accelerate 1.15.0, datasets 5.0.1, tokenizers 0.23.2, numpy 2.4.6, pandas 3.0.6, codecarbon 3.3.1, nvidia-ml-py 13.610.43, pytest 9.1.1, spaCy 3.8.16, en-core-web-sm 3.8.0.

Notes on the recipe:
- CAIR's own pin for GraphTM: python `>=3.11,<3.12`, CUDA `>=12.9.1,<13`, `pycuda >=2025.1.2,<2026` [verified in source: `third_party/GraphTsetlinMachine/pyproject.toml`]. This machine matches: Python 3.11.16, toolkit 12.9, pycuda 2025.1.3.
- `.venv-tm`: pin `numpy>=2.3.5,<3` FIRST, together with setuptools and wheel, then build pycuda with `--no-build-isolation`; tmu from git at commit 5605ff07 (built from source; the 0.8.3 wheel breaks under NumPy 2, see V2), GraphTM vendored and installed editable (`-e`), the hierarchical package from git pinned to commit c85ca4e7, `cupy-cuda12x` 14.x, numba, z3-solver, `python-sat==1.8.dev17` (the design decisions ask for an explicit pin), codecarbon, nvidia-ml-py (module `pynvml`), tokenizers, datasets, nltk, pytest.
- `.venv-dl` (separate processes only, results exchanged through files): `torch==2.14.0+cu126`, the LAST prebuilt torch with sm_70 (2.11 dropped Volta from cu128 builds; from 2.15 no CUDA 12.x wheels are published) [verified in source: PyTorch issue 172351 and dev-discuss notice of 2026-09-04]; a hard ceiling, never upgrade it; REQUIRE `sm_70` in `torch.cuda.get_arch_list()` (it is present [measured: `results/_bootstrap/dl_env_install.log`]); transformers, accelerate, datasets, codecarbon, pytest; spaCy 3.8 with the `en_core_web_sm-3.8.0` wheel.
- nvcc and curand must be on PATH and LD_LIBRARY_PATH (`/usr/local/cuda/lib64` is already set). Compile for sm_70 only. Never move to CUDA 13: it drops Volta.
- There is no cmake or ninja: the C++ engine is built with plain `g++` by a small `build.py`, and KenLM is impractical, so the Kneser-Ney baseline is in-house (docs/06).

## V5. Datasets [verified in source unless noted]
`datasets` 4 and later removed loading scripts (installed: 5.0.1), so only parquet/CSV/JSON repositories load. The ids below replace the initial design's script-based ids [stated: design notes; each id is confirmed by its first logged load].
- WikiText-103: `datasets.load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")` (raw; this is the one we use) [measured: `results/_bootstrap/wikitext_prewarm.log`, train 1801350 rows, validation 3760 rows, test 4358 rows]. Official counts from the Salesforce dataset page: 28,475 / 60 / 60 articles; 103,227,021 / 217,646 / 245,569 tokens (train / valid / test); vocabulary 267,735; OoV 0.4% [verified in source: Salesforce dataset page, 2026-09-22]. The raw variant keeps the tokens before <unk> replacement.
- Penn Treebank: `FALcon6/ptb_text_only` FAILS under datasets 5.0.1 (it ships `ptb_text_only.py`). Use `load_dataset("ptb-text-only/ptb_text_only", revision="refs/convert/parquet")` (measured OK) or download its parquet files and hash them [measured: `results/20260922T125436Z_prior_art_check/C_reference_and_tool_verification.md`]. Word-level with rare words as <unk>. The initial design's torchtext route is not used (no torchtext here).
- R8, R52: `dxgp/R8`, `dxgp/R52`. tmu ships no loader for them (V2).
- IMDB: `stanfordnlp/imdb`.
- BLiMP: `nyu-mll/blimp`, 67 configs of 1,000 pairs each, exposed as ONE split named `train` [measured, 2026-09-22]; report per-phenomenon and aggregate; human 96.4% [verified in paper].
- CondaQA: `lasha-nlp/CONDAQA`; JSON files, splits train 5,832 / validation 1,110 / test 7,240 [measured, 2026-09-22]; 14,182 QA pairs, 200+ negation cues; consistency metric; human 81.58% [verified in paper / source].
- NaN-NLI: `joey234/nan-nli`; 258 sub-clausal negation premise-hypothesis pairs exposed as split `train` [measured, 2026-09-22] (arXiv 2210.03256) [verified in paper].
- LAMBADA: `EleutherAI/lambada_openai`; held-out final word predictable from full context [verified in paper: Compressive Transformer paper description].
- PPDB (paraphrase database) and WordNet (via nltk `wordnet`) for synonym purity of exported OR groups [stated].

## V6. Tooling [verified in source unless noted]
- spaCy dependency parsing (in `.venv-dl` only): `nlp = spacy.load("en_core_web_sm")`; `token.dep_ == "neg"` marks a negation modifier on its head; `token.head` gives the governor; note en_core_web_sm attaches `neg` to the auxiliary/verb, which can be ambiguous, so resolve scope with a DependencyMatcher pattern on `{"DEP": "neg"}` plus governor propagation [verified in source: spaCy issue 2277 and DependencyMatcher example]. Parser-derived features are an ablation computed from MASKED text only (docs/06, leakage rule).
- Hugging Face tokenizers BPE at 8k to 16k: `from tokenizers import ByteLevelBPETokenizer; t = ByteLevelBPETokenizer(); t.train(files, vocab_size=8000, min_frequency=2)` (or `Tokenizer(models.BPE())` with `trainers.BpeTrainer(vocab_size=16000)`) [verified in source: tokenizers library]. Train on the train split only.
- z3 and PySAT for SAT receipts: `from z3 import Bool, And, Or, Not, Solver`; `from pysat.solvers import Glucose3` and `from pysat.formula import CNF` [verified in source: library docs].
- Profiling: `nsys` is NOT installed on this machine, so the initial design's `nsys profile` commands cannot run. `nvprof` 12.9.79 is present, as is `compute-sanitizer`. Exact support: the CUDA 12.9 profiler guide says "The NVIDIA Volta platform is the last architecture on which these tools are fully supported"; on compute capability 7.5 and later nvprof traces only and collects no events or metrics [verified in source: NVIDIA docs and forum, 2026-09-22]. Use `nvprof` plus CUDA events for the kernel time split. GPU performance counters are admin-only; tracing works [stated: design notes].
- Energy: NVML GPU power is readable through `pynvml`; RAPL symlinks exist but may dangle, so CPU joules are [derived] unless M0 proves RAPL readable [stated: design notes]. CodeCarbon: `from codecarbon import EmissionsTracker; tracker = EmissionsTracker(); tracker.start(); ...; emissions = tracker.stop()`; a `@track_emissions` decorator and a `with EmissionsTracker() as tracker:` context manager are also provided [verified in source: CodeCarbon docs].
