# 11 Engine spec: sparse event-driven Tsetlin Machine for language (normative)

Status: v0.3, normative (v0.1 plus sections 9 and 10, added 2026-09-22, docs/12 F3; section 11, the count-vote read-out and the CVTM, added 2026-09-24, docs/12 F19). Three implementations must follow this text exactly and agree
bit for bit on `state_hash()`: the Python oracle (`src/tmlm/engine/oracle/oracle.py`), the
C++ host engine, and the CUDA engine (the last two share `src/tmlm/engine/core/tm_core.h`).
Where two implementations disagree, this document is ambiguous and gets fixed first.
Every default below is a starting point to be tuned in milestone M3, and every tuning
result is logged under `results/`. No number in this file is a measurement.

Conventions: all arithmetic that influences learning is integer. `u8/u16/u32/u64/i16/i32`
are fixed-width integers. "w.p. x" means a Bernoulli draw as defined in section 3. Ties are
always broken toward the lower index. "Saturating" means clamped to the type range stated.

## 1 Data model

### 1.1 Corpus store
- `tok`: `u16[N_tok]`, BPE token ids in `[0, V)`, `V <= 32768`.
- `doc_start`: `u32[N_doc + 1]`, strictly increasing, `doc_start[0] = 0`,
  `doc_start[N_doc] = N_tok`. Position `p` belongs to document `d` iff
  `doc_start[d] <= p < doc_start[d+1]` (binary search).
- `stop`: `u8[V]`, 1 for the `N_STOP = 64` most frequent training tokens, else 0.

### 1.2 Symbol space (`u16`)
| Range | Meaning |
|---|---|
| `[0, V)` | token id (kind 0) |
| `[V, V+16)` | cluster prefix, 4 bits (kind 1) |
| `[V+16, V+272)` | cluster prefix, 8 bits (kind 2) |
| `[V+272, V+4368)` | cluster prefix, 12 bits (kind 3) |
| `SYM_MASK = V+4368` | the position is hidden |
| `SYM_EDGE = V+4369` | the position lies outside the document |
| `[V+4370, V+4370+N_POL)` | polarity symbols (reserved, unused until the polarity ablation) |

`kind_map`: `u16[4][V]`, `kind_map[0][t] = t`, `kind_map[k][t]` = the kind-k cluster symbol of
token `t`. `kind_of(sym)` is determined by the range. `N_SYM = V + 4370 + N_POL <= 65535`.

### 1.3 Configuration of an example
An example is a target position `p` with target class `y = tok[p]`, in epoch `e`. Its uid is
`a = (e << 40) | p`. The model config has a `mode`:
- `masked`: both sides visible; context positions are hidden at rate `r = 0.15`.
- `diffusion`: both sides visible; rate `r_a` drawn per example, uniform on (0, 1):
  `r_a_q32 = rng32(seed, S_RATE, a, 0, 0)`.
- `causal`: only the left side exists; nothing is hidden (`r = 0`).

Hidden test: `hidden(a, q) = rng32(seed, S_MASK, e, q, 0) < r_q32` in `masked` mode (depends on
the epoch and the position only, so all targets in an epoch see the same corruption), and
`rng32(seed, S_MASK, a, q, 0) < r_a_q32` in `diffusion` mode. The target position itself is
always treated as hidden. Frozen evaluation sets replace `hidden` by a stored bitmap.

Context symbol: `ctx(a, q, k)` = `SYM_EDGE` if `q` is outside the document of `p`, else
`SYM_MASK` if `q == p` or `hidden(a, q)`, else `kind_map[k][tok[q]]`.

### 1.4 Segments and literals
- Exact segments `seg in [0, 16)`: offset `o = seg - 8` for `seg < 8` (offsets -8..-1) and
  `o = seg - 7` for `seg >= 8` (offsets +1..+8). In `causal` mode only `seg < 8` exist.
- Bag segments `seg = 16 + 2*k + side`, `k in [0, 13)`, `side` 0 = left, 1 = right. Bucket `k`
  covers distances `d` with `8 * 2^k < d <= 8 * 2^(k+1)` (so up to 65536 tokens). `N_BAG_K` (how
  many buckets are enabled) is a config value; `N_BAG_K = 0` disables bags.
- Literal id: `lit = (seg << 17) | (sym << 1) | neg`, `u32`. `LIT_EMPTY = 0xFFFFFFFF`.
- Truth of a positive exact literal `(seg, sym)` on example `a`:
  `ctx(a, p + o, kind_of(sym)) == sym`; for `sym in {SYM_MASK, SYM_EDGE}` compare against
  `ctx(a, p + o, 0)`.
- Truth of a positive bag literal `(seg, sym)`: there is a `q` on the given side with distance
  in the bucket range, inside the document of `p`, not hidden, `stop[tok[q]] == 0`, and
  `kind_map[kind_of(sym)][tok[q]] == sym`. Only kinds 0 and 2 may appear in bag literals.
- A negated literal is true iff its positive form is false.

## 2 Model state

`C` clauses. Slot arrays use `L_SLOTS = 32`, `V_SLOTS = 16`, literal budget `BUDGET = 12`.

| Array | Type | Meaning |
|---|---|---|
| `lit_id` | `u32[C][L_SLOTS]` | tracked literals, `LIT_EMPTY` if free |
| `lit_state` | `u8[C][L_SLOTS]` | TA state; INCLUDED iff `>= 128`; 0 for free slots |
| `vote_cls` | `u16[C][V_SLOTS]` | class ids, `0xFFFF` if free |
| `vote_w` | `i16[C][V_SLOTS]` | signed weight, never 0 in a used slot, range [-32767, 32767] |
| `decay_mark` | `u64[C]` | see 5.4 |
| `own_fire_mass` | `u64[C]` | see 5.4 |
| `stat_fire`, `stat_tp` | `u32[C]` | fired events seen; of which target owned (saturating) |
| `birth_step`, `last_tp_step` | `u32[C]` | bookkeeping for recycling |
| `alive` | `u8[C]` | 1 if the clause is in use |
| `G` | `u64[V]` | cumulative target gate mass per class, Q16 (sections 5.4 and 5.5; the earlier "Q32" here was an inconsistency, fixed 2026-09-22) |
| `step` | `u64` | number of completed batches |

Derived (never hashed, recomputed by `refresh(j)`): `inc_lit[C][BUDGET]` (included literals in
4 groups: exact positive, exact negated, bag positive, bag negated; ascending literal id
inside each group),
`n_inc[C]`, `anchor[C]`, the inverted index, and the free list (all `j` with `alive[j] == 0`,
ascending).

Invariants: `n_inc <= BUDGET`; a used vote slot has `w != 0`; literal ids are unique within a
clause; class ids are unique within a clause. A CLEARED literal slot is `(LIT_EMPTY, state 0)`,
a cleared vote slot is `(0xFFFF, w 0)`. The INITIAL state has every clause free (`alive = 0`),
every slot cleared, every other field 0 (`tmlm.engine.api.empty_state`). Recycling a clause
restores exactly this cleared form. `birth_step` and `last_tp_step` store `step` truncated to
32 bits.

Shared contract code (no learning logic, used by all implementations and tests):
`src/tmlm/engine/config.py` (constants, literal packing, segment tables, `EngineConfig`, the
host-side `derive()` that produces every integer threshold), `src/tmlm/engine/rng.py` (reference
randomness) and `src/tmlm/engine/api.py` (`EngineBase`, `CorpusArrays`, `TabularData`,
`STATE_FIELDS`, `state_hash_of`, `DIAG_COUNTERS`). The example set is `corpus.targets`
(`u32[N_ex]`): the epoch permutation runs over indices into it and `p = targets[perm(e, idx)]`.
All implementations must also return identical `DIAG_COUNTERS` for identical calls.

Anchor: among included positive exact literals, the one whose symbol has the smallest training
frequency `sym_freq[sym]` (static `u32[N_SYM]`; `SYM_MASK` and `SYM_EDGE` count as the maximum
frequency), ties toward the lower literal id. `ANCHOR_NONE` if there is none.
A clause FIRES on an example iff `alive`, `anchor != ANCHOR_NONE`, and every included literal is
true. `owns(j, c)` iff clause `j` has a vote slot for `c` with `w > 0`.

`state_hash()` = SHA-256 over the little-endian bytes of, in this order: `lit_id`, `lit_state`,
`vote_cls`, `vote_w`, `decay_mark`, `own_fire_mass`, `stat_fire`, `stat_tp`, `birth_step`,
`last_tp_step`, `alive`, `G`, `step`.

## 3 Deterministic randomness

```
mix64(z): z += 0x9E3779B97F4A7C15; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9;
          z = (z ^ (z >> 27)) * 0x94D049BB133111EB; return z ^ (z >> 31)      (all mod 2^64)
rng64(seed, stream, a, b, c) = mix64(mix64(mix64(mix64(seed ^ (stream * 0xD6E8FEB86659FD93)) + a) + b) + c)
rng32(...) = rng64(...) >> 32
bern(p_q32, ...) = rng32(...) < p_q32
pick(n, ...) = (rng32(...) * n) >> 32          (n < 2^32, result in [0, n))
```
Streams (`u64`): `S_MASK=1, S_RATE=2, S_GATE_TGT=3, S_GATE_NEG=4, S_TYPE1_DEC=5, S_INSERT=6,
S_INSERT_PICK=7, S_NEG_INSERT=8, S_NEG_PICK=9, S_SEED=10, S_SEED_PICK=11, S_DECAY=12,
S_VETO=13, S_PERM=14, S_SAMPLER=15, S_NEG_DRAW=16` (the last added 2026-09-22 for `neg_mode = binomial`, section 8c).
Probabilities are converted ONCE on the host: `p_q32 = min(2^32 - 1, floor(p * 2^32))`.
No draw may depend on thread ids, batch composition, grid or block sizes.

Epoch order `perm(e, x)` for `x in [0, N)`: let `bits` be the smallest EVEN number with
`2^bits >= N`, `h = bits / 2`, `m = 2^h - 1`. Repeat: `L = x >> h; R = x & m;` then for
`r = 0, 1, 2, 3`: `(L, R) = (R, L ^ (rng32(seed, S_PERM, e, r, R) & m))`; `x = (L << h) | R`;
until `x < N` (cycle walking). Batch `b` of epoch `e` consists of `perm(e, b*B) ..
perm(e, b*B + B - 1)` (the last batch may be short); the index of an example inside its batch
is its event order `i`. Kind thresholds in 5.2.3 are `floor(0.7 * 2^32)`, `floor(0.8 * 2^32)`,
`floor(0.9 * 2^32)` compared with `<` against one `rng32` draw.

## 4 Forward pass (one batch, against the state at batch start)

1. For every example build `ctx` for the 16 exact segments and the 4 kinds.
2. Candidates: for each exact segment, if `ctx(a, p+o, 0)` is `SYM_MASK` or `SYM_EDGE` look up
   the posting list of that single literal; otherwise look up the 4 literals `(seg, ctx(a, p+o, k))`
   for `k = 0..3`. Every clause in a looked-up posting list is a candidate. (Each clause is
   anchored under exactly one literal and each literal is looked up at most once, so a clause is
   a candidate at most once per example.)
3. Verify: the candidate fires iff all literals in `inc_lit` are true (4.1 order: exact
   positive, exact negated, bag), giving the fired list `F(i)`.
4. Scores: `S_i[c] = sum over j in F(i), over used vote slots with class c, of w`. `i32`.
5. Per example: `Sy = clamp(S_i[y], -T, T)`; target gate mass `Py_q32 = ((T - Sy) << 32) / (2T)`
   capped at `2^32 - 1`; `cover = #{ j in F(i) : owns(j, y) }`; negatives = up to `K_NEG = 2`
   classes `c != y` with the largest `S_i[c]`, ties toward the lower class id, considering ONLY
   classes with `S_i[c] > 0` (a class with no positive evidence is not a competitor; without
   this rule the all-zero scores at the start of training would always select class ids 0 and
   1). Their gate mass is `Pc_q32 = ((T + clamp(S_i[c], -T, T)) << 32) / (2T)`, same cap.

Prediction uses steps 1 to 4 only. Top-k order: higher score first, ties toward lower class id.

## 5 Learning (one batch)

Normative order per batch: forward pass (4) -> feedback per touched clause (5.1 to 5.4) ->
gate mass update (5.5) -> seeding (5.6) -> `refresh` of changed clauses and index rebuild ->
every `SWEEP_PERIOD = 256` batches: sweep (5.7) -> `step += 1`.

### 5.1 Events (R1)
A clause is TOUCHED if it appears in some `F(i)`. For a touched clause `j`, first settle its
decay (5.4) using `G` as of batch start. Then process its events in ascending `i`, at most
`E_CAP = 256` (the rest are dropped and counted in a diagnostic). For event `(i, j)`:
re-evaluate "fires" against the LIVE state of `j` (earlier events of this batch may have changed
it). If it no longer fires, skip. Otherwise `stat_fire += 1`, and if `owns(j, y)`:
`stat_tp += 1`, `last_tp_step = step`, `own_fire_mass[j] += Py_q32`. Then apply 5.2 and 5.3.

### 5.2 Target class (R2)
Gate: `bern(Py_q32; seed, S_GATE_TGT, a, j, 0)`. If it fails, nothing. If it passes, let `w` be
the vote of `j` for `y` (0 if there is no slot):
- `w > 0`: Type Ia (5.2.1), then `vote_add(j, y, +1)`.
- no slot: `vote_add(j, y, +1)` FIRST; apply Type Ia only if a slot was actually obtained.
  (Literal refinement is driven only by classes the clause really votes for. A clause whose
  vote slots are full of stronger votes ignores unrelated targets.)
- `w < 0`: Type II (5.2.2), then `vote_add(j, y, +1)`.

`vote_add(j, c, delta)`: if a slot exists, `w = clamp(w + delta, -32767, 32767)`; if `w` became 0
free the slot (and run the ownership-change protocol of 5.4 if `w` was positive). If no slot
exists: use the lowest free slot; if none is free, find the slot with the smallest `|w|`
(ties toward the lower slot index) and replace it only if that `|w| <= 1`; otherwise do
nothing. A newly created positive vote triggers the ownership-change protocol.

#### 5.2.1 Type Ia
For each used literal slot `k` in ascending order, with truth `v` on this example:
- `v` true: `state = min(255, state + 1)`, except that a transition 127 -> 128 is suppressed
  (state stays 127) when `n_inc == BUDGET`. `n_inc` is updated immediately on a crossing.
- `v` false: w.p. `1/s` (`bern(inv_s_q32; seed, S_TYPE1_DEC, a, j, k)`): `state = state - 1`;
  if it reaches 0 the slot is freed.
Insertion: w.p. `P_INS = 1/8` (`S_INSERT, a, j, 0`) sample one candidate literal (5.2.3) that
is TRUE and POSITIVE on this example. If it is already tracked, nothing. Otherwise place it at
`state = 123` in the lowest free slot; if none is free, evict the EXCLUDED slot with the
lowest state (ties toward the lower slot index), but only if that state is `< 123`.

#### 5.2.2 Type II
For each used literal slot `k` with `state < 128` whose literal is FALSE on this example:
`state += 1` (a 127 -> 128 crossing is suppressed when `n_inc == BUDGET`).
Negated insertion: w.p. `P_NEG = 1/16` (`S_NEG_INSERT, a, j, 0`) sample one exact candidate
(5.2.3, exact segments only, excluding `SYM_MASK`), take its NEGATION (false on this example)
and insert it with the same slot rule as Type Ia at `state = 123`. This is the only way a
negated literal can enter a clause.

#### 5.2.3 Candidate sampling
`u = rng32(seed, PICK_STREAM, a, j, t)` for successive `t = 0, 1, 2` (segment, kind, bag
position). Segment: exact segments have weight `2^(8 - |o|)` each, enabled bag segments have
weight 8 each; sample by cumulative weights (fixed table, host-built, mode dependent). Kind:
for exact segments token (kind 0) w.p. 0.7, else kinds 1, 2, 3 w.p. 0.1 each; for bag segments
kind 0 w.p. 0.7, kind 2 w.p. 0.3. Exact candidate = `(seg, ctx(a, p+o, kind))`. Bag candidate:
pick a distance uniformly in the bucket range, take `q`; if `q` is outside the document, hidden,
or a stop token, there is no candidate (no retry). `PICK_STREAM` is `S_INSERT_PICK`,
`S_NEG_PICK` or `S_SEED_PICK`. Segment choice: `x = pick(total, ...)`, then the first table
entry whose cumulative weight is `> x`. Type Ia insertion uses the full table (`seg_ids`,
`seg_cum`, `seg_total` of `derive()`); negated insertion and seeding are restricted to exact
segments and use the exact-only table (`exact_seg_ids`, `exact_cum`, `exact_total`). Bag
distance: `d = lo + pick(hi - lo + 1, ..., t = 2)` with `(lo, hi)` the inclusive bucket range,
`q = p - d` on the left side and `q = p + d` on the right side. A candidate whose symbol is
`SYM_MASK` is "no candidate" for every use; `SYM_EDGE` is allowed.

### 5.3 Negative classes (R3)
For the `n`-th selected negative `c` of example `i` (`n` in 0..K_NEG-1), gate:
`bern(Pc_q32; seed, S_GATE_NEG, a, j, n)`. If it passes, with `w` the vote of `j` for `c`:
- `w > 0`: Type II (5.2.2), then `vote_add(j, c, -1)`.
- `w < 0`: Type Ia (5.2.1), then `vote_add(j, c, -1)`.
- `w == 0`: if `S_i[c] >= S_i[y] - MARGIN` (`MARGIN = T/4`) then w.p. `P_VETO = 1/16`
  (`S_VETO, a, j, n`) create the vote `(c, -1)` but ONLY in a free slot (never by eviction).
`neg_mode = uniform_binomial` (used only for parity tests, small class counts) replaces the top-K
selection by the dense gate of the standard algorithm, which is an exact Binomial(K-1, Q/(K-1))
draw per clause: for clause `j` and every class `c != y` in ascending order, `c` is selected iff
`bern(q_neg_q32; seed, S_GATE_NEG, a, j, 65536 + c)` with `q_neg_q32 = q32(min(1, Q/(K-1)))`; the
error gate of a selected pair is then `bern(Pc_q32; seed, S_GATE_NEG, a, j, c)`.

### 5.4 Lazy decay scoped to owned classes (R4)
Standard Type Ib feedback decrements every literal of a non-firing clause w.p. `1/s` whenever
the clause is selected for Type I feedback. Here it is applied lazily and only with respect
to classes the clause owns. All gate masses are kept in Q16 (`mass_q16 = P_q32 >> 16`), which
cannot overflow `u64` in any realistic run (Q32 could, within tens of epochs). `G`,
`decay_mark` and `own_fire_mass` are Q16. Define `M(j) = sum of G[c] over classes c with
owns(j, c)`. `settle(j)`: `owed = M(j) - decay_mark[j] - own_fire_mass[j]` (if negative, treat
as 0); `n_q16 = owed / s` (integer division); `n_int = n_q16 >> 16`;
`frac_q32 = (n_q16 & 0xFFFF) << 16`. For each used literal slot `k` ascending:
`dec = n_int + (bern(frac_q32; seed, S_DECAY, step, j, k) ? 1 : 0)`;
`state = max(0, state - dec)`; a slot reaching 0 is freed. Then `decay_mark[j] = M(j)`,
`own_fire_mass[j] = 0`.
Ownership-change protocol (a vote for `c` becomes positive, or stops being positive): call
`settle(j)` BEFORE changing the vote, change the vote, then set `decay_mark[j] = M(j)` with the
new ownership. `decay_mode = eager_exact` (oracle only, for A/B tests) applies the per-example
per-literal Bernoulli decrements of the standard algorithm instead.

### 5.5 Gate mass
After all feedback of the batch: `G[y_i] += Py_q32(i) >> 16` for every example `i` (order free).
Likewise 5.1 adds `Py_q32 >> 16` to `own_fire_mass`.

### 5.6 Seeding (R5)
Example `i` requests a seed iff `cover < K_COVER = 2` and
`bern((Py_q32 * RHO_SEED_q32) >> 32; seed, S_SEED, a, 0, 0)` with `RHO_SEED = 1/4`.
Requests are served in ascending `i`, at most `SEED_CAP = 4` per class per batch, while the
free list is non-empty (lowest free clause id first). A new clause gets `n_lit` literals,
`n_lit = 1, 2, 3` w.p. `0.5, 0.35, 0.15` (`S_SEED_PICK, a, 0, 100`): sample exact candidates
(5.2.3 with `j = 0`, `t = 3*m .. 3*m+2` for literal `m`), rejecting `SYM_MASK` and duplicates of
an already chosen segment (a rejected draw is simply lost, so a clause may get fewer literals;
with zero literals the seed is abandoned and the clause stays free). Literals are stored in
draw order from slot 0 with `state = 136`. Vote slot 0 = `(y, +1)`. `alive = 1`,
`birth_step = last_tp_step = step`, `stat_* = 0`, `decay_mark = G[y]` (value after 5.5),
`own_fire_mass = 0`.

### 5.7 Sweep and recycling
Every `SWEEP_PERIOD` batches, for every alive clause in ascending id: `settle(j)`; `refresh(j)`.
A clause is recycled (`alive = 0`, all slots cleared, all per-clause fields zeroed) if
`anchor == ANCHOR_NONE`, or it owns no class, or `step - last_tp_step > DEAD_SWEEPS *
SWEEP_PERIOD` (`DEAD_SWEEPS = 8`). If afterwards fewer than `MIN_FREE = C/128` clauses are
free, recycle additional clauses older than `2 * SWEEP_PERIOD` in ascending order of
`(stat_tp, id)` until `MIN_FREE` are free. Finally `stat_fire` and `stat_tp` are halved.

## 6 Tabular input, dense-equivalent mode (R8) and the deviation ladder
Tabular input (for parity tests on non-language tasks): an example is a Boolean feature vector
`x in {0,1}^F` with a class label. It uses the single pseudo-segment 0, symbol `f` for feature
`f`, literal `(0, f, neg)`; the positive literal is true iff `x[f] == 1`. There is no corpus,
no hiding, no bags; the example uid is `a = (e << 40) | row`.

Dense-equivalent mode: every one of the `2F` literals pre-tracked at `state = 127` (needs
`L_SLOTS >= 2F`; slot `2f` holds the positive literal of feature `f`, slot `2f + 1` its
negation; remaining slots cleared), every clause `alive`, a vote slot for EVERY class (slot `c`
holds class `c`) initialised to `+1` if `rng32(seed, S_SEED, j, c, 0) & 1` else `-1`
(needs `V_SLOTS >=` number of classes; votes may pass
through 0 here without freeing the slot, and `w == 0` counts as `w >= 0`), no budget, no anchor
requirement (a clause with no included literal FIRES during training and does not fire at
prediction, as in the standard TM), brute-force evaluation of all clauses, no insertion, no
seeding, no recycling, `neg_mode = uniform_binomial`, `decay_mode = eager_exact`, `B = 1`.
Feedback per example then follows the standard Coalesced TM: for the target class, clauses
with `w >= 0` get Type I and clauses with `w < 0` get Type II; for each drawn negative class,
clauses with `w >= 0` get Type II and clauses with `w < 0` get Type I; each (clause, class)
pair is gated independently as in 5.2 / 5.3. Type I on a FIRING clause is 5.2.1 without
insertion; Type I on a NON-FIRING clause decrements every literal w.p. `1/s`
(`S_TYPE1_DEC, a, j, k`); Type II on a non-firing clause does nothing; votes change only when
the clause fires. In this mode the engine is a standard Coalesced Tsetlin Machine and must
reach accuracy parity with `tmu` on small tasks. The deviation ladder
(experiment s4) then enables one deviation at a time (D1 owned-class lazy decay, D2 sparse
votes, D3 bounded literal slots with insertion, D4 seeding, D5 the negative-class rule of the
default engine (section 8c, `binomial` since 2026-09-22; `focused` was the v0.1 rule) instead of
`uniform_binomial`, D6 anchor index, D7 batching) and logs accuracy and cost after each step.
The last rung is the default engine, so on the tiny LM it must replay `experiments/m3_tiny_lm.py`.

## 7 Hyperparameter defaults (all tunable, all logged with every run)
`T = 64`, `s = 128`, `L_SLOTS = 32`, `V_SLOTS = 64`, `BUDGET = 12`, `NEG_BUDGET = 2`, `K_NEG = 2`,
`K_COVER = 8`, `P_INS = 1/4`, `P_NEG = 1/16`, `P_VETO = 1/16`, `RHO_SEED = 1`, `SEED_CAP = 32`,
`E_CAP = 256`, `SWEEP_PERIOD = 256`, `DEAD_SWEEPS = 32`, `MARGIN = T/4`, `N_STOP = 64`,
`N_BAG_K = 13`, `B = 1024`, `SEG_WEIGHTING = recency`, `NEG_MODE = binomial` (section 8c), `CAND_TRIES = 1`.
History. v0.1: `s = 16`, `P_INS = 1/8`, `DEAD_SWEEPS = 8`, recency weighting, no negated-literal cap.
First change, 2026-09-22, on the planted-rules gate: `s = 32`, `P_INS = 1/4`, `DEAD_SWEEPS = 32`, uniform
weighting, `NEG_BUDGET = 2` [measured: `results/20260922T165439Z_planted_rules/`, docs/12 F15].
Second change, the same day, on the tiny LM (KB2, docs/12 F17): `s = 128`, `V_SLOTS = 64`, `K_COVER = 8`,
`RHO_SEED = 1`, `SEED_CAP = 32`, recency weighting again, `NEG_MODE = binomial`. With the planted-rules
values the tiny LM stayed below the left bigram (0.036 and 0.133 vs 0.1447); with these it reached top-1
0.1891 after 4 epochs, above the bigram and at 94% of the count probe [measured:
`results/20260922T190301Z_tiny_lm/`, seed 1; seeds 2 to 5: 0.1865, 0.1883, 0.1877, 0.1878, `results/20260922T193047Z_tiny_lm/`; H1.4 met on 5 of 5 seeds]. The reading
(docs/results.md, KB2 diagnosis, third round): text needs many general clauses that stay stable (high `s`),
a seeding rate that fills the bank, recency-weighted candidates, and negatives drawn by frequency
rather than from the top of the class sum. The planted-rules configuration keeps its own values spelled
out in `experiments/configs/planted_rules.json`, so the logged gate run still replays. Provisional
until the deviation ladder (s4) prices each of these on text.

## 8 Known open questions for M3 (to be answered by experiments, not by argument)
1. Whether `s` should differ between Type Ia false-literal decrements and owned-class decay
   (rare-pattern survival needs slow decay, over-specialisation control needs fast exclusion).
2. Whether Type Ia insertion should be rarer than 1/8 once clauses are seeded from data.
3. How much accuracy batching costs (experiment s3) and whether `E_CAP` needs to scale with B. Answered 2026-09-23 [measured: `results/20260923T020158Z_s3/`, 3 seeds]: 0.7 points from B = 64 to B = 1024 and 1.5 points to B = 4096; B = 1 is seeding-starved (question 7). Whether `E_CAP` was hit is not in the s3 metrics (the counter `n_events_dropped_cap` is not written by `train_eval`): to be recorded in the M7 runs.
4. Whether sparse votes need the wide tier for high-entropy contexts (perplexity, not top-1).
5. (Added 2026-09-22 from Zhang, Jiao, Granmo, ICLR 2026, Lemma 4 and Theorem 9.) Convergence to several sub-patterns of a class needs `T <= floor(m/2)`, or `floor(m/e)` with irrelevant bits, where `m` is the number of clauses that can vote for that class. Here `m` is the number of clauses OWNING the class, not `C`; with `T = 64` that asks for at least about 128 owning clauses per class. Whether seeding (`K_COVER`, `SEED_CAP`) reaches that for frequent classes, and what happens to rare classes that never do, is measured in s4 (clauses owned per class logged against `T`).
6. (Same source, "Insights for practical usage".) Irrelevant literals accumulate when many clauses are used; `BUDGET = 12` is the spec's answer, and s4 logs the mean number of included literals per clause per rung.
7. Seeding cap and batch size (2026-09-23, from s3): `SEED_CAP` is per class per batch, so at
   B = 1 the bank never fills (7,064 alive of 65,536 after 4 epochs vs 65,3xx at B >= 64
   [measured: `results/20260923T020158Z_s3/`]) and the B = 1 accuracy (0.126) is a seeding-rate
   artefact, not a staleness measurement. Should the cap be per example (or per token budget) so
   that batch size and seeding rate are independent? Not changed; open for the M7 scaling runs.
8. Sparse tabular mode does not learn (2026-09-23, from H2 row d and a synthetic check; docs/results.md,
   H2 row d, seed 1): seeding on unordered bit vectors draws uninformative literals and the
   duplicate-segment rule of 5.6 leaves tabular seed clauses with one literal. Rows H2 b, c, d and
   H3 need either the dense-equivalent mode on a reduced feature set or a tabular seeding rule
   (several literals, class-conditional weighting). Decision pending; nothing changed here.

## 8a Clarifications that the oracle had to decide (normative since 2026-09-22)

The oracle's docstring (`src/tmlm/engine/oracle/oracle.py`, notes N1 to N12) records every
place where the text above was silent or ambiguous. Those readings are now part of the spec;
the C++ and CUDA engines implement them as written there. In short: the sweep runs at the end
of the batch whose one-based number is a multiple of `SWEEP_PERIOD`, ages use 32-bit wrapping
subtraction (N1); `E_CAP` cuts the event list to its first `E_CAP` entries and skipped events
keep their place (N2); the target and negative steps of an accepted event both run (N3);
`SEED_CAP` counts served requests, abandoned or not (N4); in `uniform_binomial` mode the veto
draw uses the class id as `n` (N5); `owed / s` is an exact rational floor (N6); `G`,
`decay_mark` and `own_fire_mass` are Q16 (N7); tabular mode outside dense-equivalent mode uses
segment 0 literals, equal symbol frequencies and `f = pick(F, ..., t + 2)` (N8); dense-equivalent
mode frees a slot at state 0 with no way back, keeps no sweep, and needs `B = 1` (N9);
`eager_exact` decay in the sparse engine draws the target gate for every alive non-firing
owning clause (N10); a literal on a disabled segment has a FALSE positive form and polarity
symbols never match (N11); the `DIAG_COUNTERS` are defined as in N12.

## 8b Two knobs added on 2026-09-22 after the first planted-rules runs (normative)

Both are `EngineConfig` fields, implemented identically in the oracle and in `tm_core.h`, and
both keep every earlier hash unchanged at their former behaviour (`seg_weighting = recency`,
`neg_budget = -1`).
- `seg_weighting`: the candidate-sampling weights of 5.2.3. `recency` is the table above
  (`2^(8 - |o|)` per exact segment, 8 per bag segment); `uniform` gives every enabled segment
  weight 1. Reason: with recency weights a rule at offset -8 is sampled 128 times less often
  than one at -1, so seeds and insertions almost never reach it (docs/results.md, planted rules).
- `neg_budget` (N13): a separate cap on the number of INCLUDED negated literals per clause.
  The 127 -> 128 crossing of a negated literal is suppressed while the clause already includes
  `neg_budget` negated literals; `n_inc_neg` is a derived per-clause counter kept exactly like
  `n_inc` (updated on crossings in both directions, recomputed by `refresh`). `-1` disables the
  cap. Reason: Type II inserts the negation of a literal seen on a negative example; that
  negation is then true on almost every positive example, is included within a few Type Ia
  events, and never leaves, so clauses fill their `BUDGET` with negations (the "irrelevant
  literals accumulate" effect of Zhang, Jiao, Granmo, ICLR 2026, section 8) and the literals
  that matter are locked out.
The defaults in section 7 were updated from the logged run `results/20260922T165439Z_planted_rules/` (docs/12 F15): `uniform` and `NEG_BUDGET = 2` are now the defaults.

## 8c Negative classes revisited on text (2026-09-22): `neg_mode = binomial` (N14)

The first tiny-LM run failed KB2 at unigram level (docs/results.md, tiny LM, first run). Diagnosis:
the focused rule R3 selects the two highest-scoring wrong classes on every example, which under a
Zipf target distribution are the frequent classes nearly every time, so their votes are pushed
down on almost every example and reinforced only when they are the target; the model ends below
the unigram. With the negatives switched off or drawn uniformly the same engine doubled its
accuracy (exploratory runs; the logged diagnosis runs are cited in docs/results.md).

`neg_mode = binomial` implements the uniform draw of `uniform_binomial` mode at O(k) cost per
event instead of O(V): `k ~ Binomial(V - 1, Q / (V - 1))` by inversion of a host-computed Q32
CDF table (`neg_cdf_q32`, `NEG_K_MAX = 32`, tail mass negligible at Q <= 1 [derived]) with
`u = rng32(seed, S_NEG_DRAW, a, j, 0)`, then `k` distinct classes `c != y`, attempt `t` drawing
`pick(V - 1; S_NEG_DRAW, a, j, 1 + t)` (a value `>= y` maps to `c + 1`; a duplicate is rejected
and `t` advances; at most `8 * NEG_K_MAX` attempts). Each selected class then passes the error
gate `bern(Pc_q32(S_i[c]); S_GATE_NEG, a, j, c)` and is updated as in 5.3 with `n = c`. The
selected SET has exactly the distribution of independent Bernoulli(Q / (V - 1)) per class
truncated at `NEG_K_MAX`, which is the dense Q gate of the standard algorithm (this is the exact
Binomial reformulation of docs/02, now inside Track B); `tests/test_engine_knobs.py` checks the
count against Binomial(V - 1, Q / (V - 1)) by a chi-square test and the oracle against the C
engine bit for bit. It is not bit-identical to `uniform_binomial` (different draws), only
distributionally identical. Whether it becomes the default is decided by the logged tiny-LM
runs (docs/12).

## 8d Candidate quality (2026-09-22): `cand_tries` (N15)

Seeds and insertions of 5.2.3 draw ONE literal from the example's context; on text most of the
64 (segment, kind) choices are coarse cluster symbols or far offsets, so seeded clauses fired on
uninformative literals (the most-firing clause of the first tiny-LM diagnosis was
`t[+8] = c4:1`). `cand_tries = k` draws `k` candidates (try `i` uses the base draw index
`t + 9 * i`, so the three draws of a try never collide with those of another try or of another
seed literal) and keeps the one whose symbol has the SMALLEST training frequency (`sym_freq`,
the anchor principle of section 2 applied at creation time), ties toward the earlier try.
`k = 1` is the previous behaviour. Implemented identically in the oracle and `tm_core.h`;
`tests/test_engine_knobs.py` checks bit-exactness and that the kept symbol is never more
frequent than the single draw's.

## 9 Clause sharding (normative): what a multi-device run exchanges

Clauses are independent voters, so a model of `C` clauses may be trained as `S` shards, shard
`r` owning the contiguous id range `[r*C/S, (r+1)*C/S)`. Sharded training must produce, for
every shard, exactly the state the unsharded engine holds for those ids (`state_hash` of the
concatenated shard states equals the unsharded `state_hash`). This section lists everything a
shard needs from the others; a correct implementation exchanges nothing else. Experiment s5
(docs/05) tests it with 2 and 4 shards; on one GPU the shards are separate engine instances
with an explicit exchange step between phases.

Per batch, in the normative order of section 5:
1. Forward (section 4): each shard evaluates its own clauses and produces partial scores
   `S_i^r[c]` (sparse) and partial cover counts `cover_i^r`. EXCHANGE 1: every shard receives
   the sums `S_i[c] = sum_r S_i^r[c]` and `cover_i = sum_r cover_i^r` for its batch. Gate
   masses, the top-K negatives and `Sy` are then computed identically everywhere from the
   summed scores. `G` is a pure function of the examples (5.5), so every shard keeps an
   identical copy of `G` with no exchange.
2. Feedback (5.1 to 5.4): touches only the clause's own slots and reads `G`. No exchange.
3. Gate mass (5.5): identical on every shard. No exchange.
4. Seeding (5.6): requests are served in ascending `i` from the GLOBAL free list, lowest id
   first, with `SEED_CAP` per class per batch. EXCHANGE 2: the number of free clauses in every
   shard (`S` integers). Shard `r` serves the requests whose global rank falls inside its own
   free-list range; the per-class cap is computed identically everywhere from the ordered
   request list.
5. Sweep (5.7): recycling by `anchor`, ownership and age is local. The `MIN_FREE` top-up
   recycles the globally smallest `(stat_tp, id)` among clauses older than `2 * SWEEP_PERIOD`
   until `MIN_FREE` are free. EXCHANGE 3: each shard's sorted candidate list of `(stat_tp, id)`
   truncated to `MIN_FREE` entries; every shard merges the lists and recycles those of its own
   ids that fall within the first `MIN_FREE - free_total` merged entries.
6. `refresh` and the index rebuild are local: the anchor index is a per-shard structure and
   candidates per example are the union over shards (each clause appears in exactly one).

Inference: partial scores summed (exchange 1 only). Receipts list fired clauses by global id.
Nothing in the exchanges depends on `S`, on thread layout or on batch composition, so the
determinism rule of section 3 is preserved. Communication volume per batch is bounded by
`B * (K_NEG + 1 + |fired classes|)` score entries plus `S` integers plus `S * MIN_FREE`
pairs [derived].

## 10 Per-token cost bounds (what is linear in what)

Every quantity below is bounded by a constant of section 7 or is linear in exactly one of
`C` (clauses), `N` (context length), `V` (vocabulary). The constants are measured by
experiment CM (docs/05) and the bounds are what the scalability claim rests on (docs/07, P3).
No number here is a measurement.

| Step | Work per example | Depends on |
|---|---|---|
| ctx build (4.1) | 16 exact segments x 4 kinds | none |
| candidate lookup (4.2) | at most 64 posting-list lookups; candidates = sum of list lengths | C (through list lengths; KB4 bounds this) |
| verify (4.3) | at most `BUDGET = 12` literal tests per candidate; a bag literal costs one binary search, O(log N) | C, log N |
| scores (4.4) | at most `V_SLOTS = 16` adds per fired clause | fired clauses |
| gates, negatives (4.5) | top-K over the touched classes | fired clauses |
| feedback (5.1 to 5.4) | at most `E_CAP = 256` events per touched clause, each over at most `L_SLOTS = 32` slots | fired clauses |
| seeding (5.6) | at most `SEED_CAP` per class per batch | none per token |
| sweep (5.7) | O(C) every `SWEEP_PERIOD` batches: `C / (SWEEP_PERIOD * B)` clause visits per example amortised | C |
| index rebuild | O(changed clauses + C) every sweep, amortised as above | C |

Memory: about 290 B per clause [derived, docs/03] plus `O(V)` for `G`, `kind_map` and
`sym_freq`, plus the corpus. No `V x C` array exists anywhere. No step touches all `V`
classes per token during training; the full-vocabulary softmax of `logprobs` (O(V) per token)
is an evaluation-time operation for perplexity only, not a training cost. There is no
quadratic term in N: exact segments are array lookups and bag literals are verified by binary
search on an occurrence list (training) or looked up in per-bucket last-occurrence tables
(streaming inference, O(1) membership, O(log N) maintenance per token).


## 11 Count-Vote Tsetlin Machine (CVTM): the count-vote read-out, document clauses and the LM built on them (normative since 2026-09-24)

Added by docs/12 F19 (decision of 2026-09-24) after the evaluation in
`docs/results.md`. Sections 1 to 10 are unchanged: the signed-vote engine
stays the Tsetlin-automata (TA) learner and the reproduction path of every earlier number. This
section defines a different READ-OUT and two enumerated clause families. The CKM design
of 2026-09-23 is not implemented. No number in this section is a
measurement; the hypotheses are docs/13 amendment A3.

### 11.1 Model

A CVTM predicts the next token from the class sum of the votes of the clauses that fire at the
position, over the literal space of section 1. Three clause families exist:

- S, suffix clauses (enumerated): contiguous exact left contexts, the n-gram special case of a clause.
- D, document clauses (in-context memory): clauses whose read-out is the document so far.
- L, learned clauses (optional): the bank that sections 1 to 10 train with Tsetlin automata.

Votes come from exact integer counts, never from +-1 steps. "CVTM-core" is S plus D; "CVTM-full"
adds L. Code: `src/tmlm/cvtm/`.

### 11.2 Suffix clauses and the exact vote form of Kneser-Ney

Reference estimator: `KNModel` of `src/tmlm/eval/baselines/ngram.py`, order K = 5, `pad = True`,
modified discounts, fitted on the counting documents. For a context u of length k (1 <= k <= K-1)
that occurs in training, clause S(u) fires at position p iff the k tokens left of p (BOS-padded at
the document start, as in `KNModel.contexts`) equal u. Level k+1 of the recursion is
p_{k+1}(w | u) = alpha(u, w) + gamma(u) p_k(w | u'), with u' the length k-1 suffix of u. Votes:

- follower vote, for every w with a positive count after u at that level:
  v(u, w) = ln(1 + alpha(u, w) / (gamma(u) p_k(w | u')));
- default vote, for every other class: 0, plus the clause constant d(u) = ln gamma(u);
- bias clause (always fires): b(w) = ln p_1(w).

Then ln P_KN(w | h) = b(w) + sum over fired S(u) of [d(u) + v(u, w)], exactly (the interpolation
recursion telescopes), and the normaliser is known in closed form: no sum over V is needed to score
the gold token. A context unseen in training fires no clause, which is KNModel's gamma = 1 case.
Integer form: b, d and v are stored as int32 in units of 2^-16 nats (round to nearest, ties to
even); the float64 form is the reference for every reported number, the integer form is used by
receipts and must change validation bits per byte by at most 0.001 (A3, H8.2).

### 11.3 Document clauses

For m in 0..4, clause D_m fires at p iff the m-token left context c_m(p) (m = 0: the empty context)
occurred at an earlier position q of the same document with p - N <= q < p (window N; default the
whole document). Its read-out is the document-local successor distribution
P_Dm(w | p) = #{q : c_m(q) = c_m(p), tok[q] = w} / #{q : c_m(q) = c_m(p)}, over the same q range.
The pointer clause P fires iff the longest L in 5..64 such that the L tokens left of p also occur
ending just before some earlier q in the window exists; its read-out puts all mass on tok[q] of the
most recent such q. Counts are exact integers; per position the cost is O(1) hash-table work per
order (streaming) or O(log n) (sort-based, batch), independent of N.

### 11.4 Combination (fixed estimator)

P(w | p) = lambda_S(b) P_KN(w | h) + sum over firing D_m of lambda_m(b) P_Dm(w | p) + lambda_P(b) P_P(w | p),
where the weights of non-firing components are dropped and the rest renormalised to sum to one.
Bucket b(p) = (k*, m*, c, l): k* in 0..4 the longest fired suffix context; m* in {none, 0..4} the
highest firing document order; c the count bucket {1, 2-3, 4-7, 8-15, 16+} of #{q} at order m*;
l the pointer bucket {none, 5-8, 9-16, 17+}. Weights are fitted by MAP-EM (50 iterations from
uniform, float64, fixed order, Dirichlet prior of 16 pseudo-counts toward the pooled weights of all
buckets) on the fit documents of 11.6 and are frozen before any validation number is computed.
In vote form the D and P clauses add the residual vote r(w) = ln(lambda_S + sum_i lambda_i P_i(w) / P_KN(w | h))
to the class sum; the result is normalised exactly.

### 11.5 Count-vote read-out of a TA-learned bank (family L)

- Counting: one pass over the counting positions with the bank frozen; O_jc = number of positions
  where clause j fires and the target is c; N_j = sum over c. Exact integers, sort-based, on the GPU.
- Stand-alone read-out ("T1"): class score s(c) = class_bias(c) + sum over fired j of
  alpha_g(j) q_jc with q_jc = ln(1 + lam_j O_jc / (N_j (1 - lam_j) P_uni(c))), lam_j the Witten-Bell
  weight of clause j, and alpha per group g(j) = (n_inc bucket, floor(log2 N_j)); alpha fitted on the
  fit documents; softmax over V. This is the engine's own sum-of-votes form with count-derived votes
  (`loglin` of `experiments/ckm_posthoc_probe.py`).
- Residual read-out on top of CVTM-core ("T2"): P(w) proportional to P_core(w) exp(sum over fired j of alpha_g(j) v_jw)
  with v_jc = ln((O_jc + 0.5) / (E_jc + 0.5)), E_jc = sum of P_core(c | p) over the positions where j
  fires (P_core cross-fitted on the other half of the counting documents), the 64 classes of largest
  |O_jc - E_jc| kept per clause; normaliser Z = 1 + sum over classes with a vote of P_core(c)(e^s_c - 1).

### 11.6 Fitting protocol, determinism, sharding

- Fit documents H: training documents i with splitmix64(i XOR 0xC0FFEE) mod 100 == 0. A KN-5 fitted
  WITHOUT H supplies P_KN on H for the EM of 11.4; the final model uses the KN-5 fitted on ALL
  training documents, which is identical to the comparator of `results/20260922T180823Z_M2_baselines_ngram_v8000/`.
  Validation reports; test once, after the model is frozen.
- All counts are exact integers. Sharded counting (contexts partitioned by a hash of the context
  into S shards, counted independently, concatenated and re-sorted) must equal unsharded counting bit
  for bit. The EM and the float64 scoring run in a fixed order and are deterministic on one platform.

### 11.7 Cost and memory bounds (what the P3 claims for the CVTM rest on)

| Step | Work | Depends on |
|---|---|---|
| fit (counting) | one sort per order over the training stream | linear in training tokens (n log n sort) |
| gold probability | K - 1 suffix lookups + 5 document orders + pointer + one bucket lookup | none of V, N, C |
| full distribution (sampling) | the followers of the fired suffix clauses plus the document candidates | followers, not V times context |
| memory, model | keys and counts of the n-gram types plus one gamma per context | distinct n-grams (prunable) |
| memory, document | the counts of the current document window | N |

### 11.8 Masked variant (for docs/13 A3, H8.11)

The masked CVTM-core replaces the suffix clauses by the ten two-sided contiguous patterns of
`experiments/ckm_posthoc_probe.py` (left 1 to 3, right 1 to 3, and the four joint left-right
patterns), each an enumerated clause whose read-out is its exact successor distribution over the
counting positions (hidden positions cut a pattern, as in `KNModel.contexts`). The combination is the
normalised linear mixture `lin` of that script (weights per (pattern, floor(log2 N)) bucket fitted by
EM on the fit documents) plus document clauses keyed on the same patterns (the most recent earlier
visible position of the document with the same pattern votes for its token, confidence per pattern
from the fit documents).
