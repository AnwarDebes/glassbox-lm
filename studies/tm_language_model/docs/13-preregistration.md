# 13 Pre-registration of hypotheses, thresholds, seeds and analysis

Status: FROZEN on 2026-09-22 (v1). The body of this file is never edited after freezing.
Later changes go into the "Amendments" section at the end, each dated, each with its reason,
so the history is visible. `scripts/run_logged.py` records the sha256 of this file as
`prereg_sha256` in every `run.json` (protocol rule 16), so every reported number can be
matched to the version of this file that was in force when it was produced. No run that
feeds a reported number may start before this file exists.

Publication note: the wording of this copy was edited for publication on 2026-10-10 and 2026-10-11. The
registered hypotheses, thresholds, seeds, analysis plans, amendments and dates are unchanged.
The `prereg_sha256` values in the run records refer to this file as it stood when each run
started, not to this copy. The last registered revision is recorded by 384 of the 643 run
records of this study and by all 319 run records of the hierarchical_tm study; it is kept by
the author, and its SHA-256 is listed in the
[hash table](../../../docs/hash-table.md) next to that of this copy. Ten earlier revisions,
recorded by 249 run records of this study, were not retained, and 10 run records hold no such
value.

No number in this file is a measurement. Every threshold below is a target fixed BEFORE the
first run. Missing a target is reported as a negative result with its log path (protocol
rules 2 and 3, docs/08 "Positive-by-construction check").

## 1 What is being proved

The proof of concept has one claim with three pillars (docs/07, "Proof-of-concept criteria"),
plus a comparison that answers the supervisor's project proposal (unpublished; docs/07 "Answering the
proposal"):

- Claim: a Tsetlin Machine can be the core of a language model trained at corpus scale, with
  a reported likelihood, on the sparse event-driven engine of docs/11.
- P1 Accuracy: the model beats the strongest like-for-like classical comparator and is placed
  against a neural comparator at equal cost.
- P2 Reproducibility and extensibility: one bootstrap command from a clean checkout, bit-exact
  replay, a standalone receipt checker, a CPU engine that needs no GPU.
- P3 Scalability with compute: measured curves across decades of clause count and context
  length, with the per-token cost bounds of docs/11 section 10 and the clause-shard
  equivalence of docs/11 section 9.
- Proposal answer: the hierarchical TM of the project proposal, measured as a row next to the engine on
  the project proposal's own task (docs/05, H1 and H2).

## 2 Hypotheses and thresholds

Every hypothesis names the experiment (docs/05), the split, the metric, the comparator, the
threshold, and what a miss means. "bpb" is bits per byte (docs/05, protocol CP).

| Id | Pillar | Hypothesis | Experiment | Metric and split | Threshold (target) | If missed |
|---|---|---|---|---|---|---|
| H1.1 | P1 | The causal TM beats the in-house Kneser-Ney 5-gram on identical tokens | CP (M7) | bpb, WikiText-103 test, 3 seeds at the reported size | TM bpb below KN-5 bpb for every seed | KB7 path: wide tier, then "first measurement, no superiority claim" |
| H1.2 | P1 | The masked TM beats the left bigram and CBOW-kNN | E3 (M6, Gate 1) | top-1 on the frozen eval mask set, validation then test once | above both comparators, 3 seeds at V=8k | Gate 1 fails: report, diagnose, no scaling claims above it |
| H1.3 | P1 | At equal V100 wall-clock and equal joules, the causal TM is placed against a from-scratch causal transformer | TB (M7) | bpb on identical tokens and bytes; both models trained under the same wall-clock and joule caps | no threshold: the number is reported in both directions | nothing to miss; the comparison is reported as measured |
| H1.4 | P1 | The tiny LM beats the left bigram and reaches at least 70% of count-probe top-1 | s4 (M3, KB2) | top-1, tiny LM validation, 5 seeds | as stated | KB2 path (docs/08) |
| H2.1 | P2 | `env/bootstrap.sh` builds both environments from a clean checkout with its output logged | M2 step 0 | log under `results/` | one command, exit 0 | the recipe is fixed until it does; nothing else is reported before |
| H2.2 | P2 | Training and generation replay bit-exactly | M4 (engine), M8 (generation) | equal `state_hash` and byte-identical samples across two runs and two launch configurations | equality | KB6 path; no replay claim |
| H2.3 | P2 | The CPU engine reproduces the GPU engine hash | M4 | `state_hash` after 100 steps at B in {64, 1024, 4096} | equality | KB6: the GPU becomes inference-only |
| H2.4 | P2 | Every receipt of the pre-registered generations passes the standalone checker; tampered receipts fail | E11 (M8) | pass fraction | 100% pass, 100% of tampered fail | Gate 3 fails: report |
| H3.1 | P3 | Accuracy does not fall as clause count grows across the measured decades | s1 (M7) | top-1 and bpb vs C from 64k upward, 3 seeds at small sizes | non-decreasing top-1 within seed range at each doubling; a plateau is allowed | report the curve as measured; the "scales with clauses" claim is limited to the monotone range |
| H3.2 | P3 | Cost per token is at most linear in clause count | s1, CM (M7) | log-log slope of ns per token vs C over the measured range | slope at most 1.0 | report; KB4 path if candidates explode |
| H3.3 | P3 | Cost per token is nearly independent of context length | s2, CM (M7) | ns per token at N=65536 vs N=16, same clause budget, training and streaming inference | at most 2x | report the curve; the near-constant claim is withdrawn, the measured slope is stated |
| H3.4 | P3 | Longer context helps | s2 (M7) | top-1 and bpb vs N | bags add at least 0.3 points top-1 or 2% perplexity at N=4096 (KB5) | KB5: negative result reported, cost curve kept |
| H3.5 | P3 | Clause-sharded training is bit-exact with unsharded training | s5 (M3 oracle, M4 GPU) | `state_hash` of concatenated shards vs unsharded, 2 and 4 shards, 200 steps | equality | the exchange of docs/11 section 9 is incomplete: fix the spec; no multi-device claim |
| H4.1 | proposal | The hierarchical package learns the project proposal's paragraph-pair task above chance at reduced vocabulary | H1 (M9) | accuracy, pair validation then test, 5 seeds | above 50% by more than the seed range | reported as measured; the row stays in the table |
| H4.2 | proposal | Engine layer 1 clause bits beat the hierarchical package on the same task | H1 vs H2 (M9) | accuracy, same pairs, 5 seeds, paired Wilcoxon | one-sided p below 0.05 | reported either way |
| H4.3 | proposal | A greedy layer 2 over layer-1 fired-clause bits improves on layer 1 | H2 (M9) | accuracy delta on the paragraph task and top-1 delta on the tiny LM, 5 seeds, paired Wilcoxon | one-sided p below 0.05 | negative result reported with the delta and its interval |
| H5.1 | kill | Stock GraphTM E1 beats the left bigram | E1 (M1) | masked top-1, V=2000 subset | above the left bigram | kill criterion of the initial design: stop and diagnose |
| H5.2 | kill | The class loop dominates the stock step time | T0.2 (M1) | nvprof fraction | at least 40% | drop the sparse-draw headline, keep it as engineering |
| H5.3 | kill | The exact Binomial draw is distributionally identical and at least 2x faster at Q=1 | E2 (M5) | the four assertions of docs/02; examples/s | pass, 2x | keep dense, drop the headline |

## 3 Seeds

- Seed list, fixed: `1, 2, 3` for every 3-seed configuration; `1, 2, 3, 4, 5` for every
  5-seed configuration. No other seeds are used for reported numbers. Exploratory runs use
  seeds 100 and above and are never reported.
- Where a configuration is run once (the largest sizes), the seed is 1 and the number is
  labelled "single-seed".
- Track A: `np.random` is seeded before importing GraphTM and `(seed, grid, block)` is logged.
  Track B: counter-based RNG (docs/11 section 3); the seed is the `seed` field of the config.
- Generation prompts: the first 20 test documents of WikiText-103, seeds 1 to 5, temperature
  fixed in the config; every sample is logged (docs/06).

## 4 Splits, tokenization and hashes

- Data: WikiText-103 raw (`Salesforce/wikitext`, `wikitext-103-raw-v1`); the standard train,
  validation and test splits, unchanged.
- Tokenizer: byte-level BPE, V=8000 and V=16000, trained on the train split only.
- Cluster codes: trained on the train split only.
- Eval mask sets: frozen, subword and whole-word variants, hashed.
- Paragraph-pair task (H1, H2): pairs are built inside one split only, so no test document
  appears in a training pair; balanced 50/50; negatives drawn from other documents of the
  same split; the pair list is frozen and hashed.
- The sha256 of every artifact above is written by milestone M2 into
  `docs/notes/preregistration-data-hashes.json` (a file that is written once and then never
  edited) and into every run's `dataset_hashes`. This file does not contain them, so its own
  hash is stable from freezing.
- Test-set access budget: one access per reported configuration, after every choice has been
  frozen on validation. The access counter of M2 is reported next to every test number.

## 5 Analysis plan

- Primary comparisons are paired across seeds. With 5 seeds the paired one-sided Wilcoxon
  signed-rank test can reach p = 1/32 [derived]; with 3 seeds it cannot reach 0.05, so
  3-seed configurations report mean, min and max only and make no significance claim.
- Every table reports mean and the min to max range across seeds; single-seed rows are
  labelled.
- Throughput and energy: median of at least 5 repeats under the exclusive GPU lock, with the
  min to max range; idle power subtracted and reported.
- Curves (s1, s2, CM): log-log plots of accuracy and cost against C or N, one line per seed.
  Slopes are least-squares fits over the measured points only. Any extrapolation beyond the
  measured range is drawn dashed, labelled [derived], and never used in a claim.
- Contamination: every test number is reported on the full test set and on the de-overlapped
  subset (docs/06).
- Negative results, triggered kill criteria and missed targets are reported in the same
  tables as positive ones, with their log paths.

## 6 What is NOT claimed

- No claim about parameter counts or context lengths beyond the measured range. The largest
  measured clause count and context length are stated with every scaling statement.
- No claim that the engine matches a transformer at scale. H1.3 is a placement at equal
  cost on one V100, reported in both directions.
- No human-quality claim for generated text.
- "First TM language model" is not claimed; the qualified claim of docs/07 is.

## Amendments

### A1 (2026-09-22, same day as freezing): speed of the model as a pre-registered property
Reason: it was made a requirement that the resulting model be fast and that speed be treated as an important factor. Added hypotheses; nothing above is changed.

| Id | Hypothesis | Experiment | Metric | Threshold (target) | If missed |
|---|---|---|---|---|---|
| H6.1 | At batch 1 on the 6-CPU engine, the TM needs less time per token than the from-scratch causal transformer on the same 6 CPUs, same positions | E10, CM (M7) | ms per token, median of 5 repeats under the exclusive lock | TM below the transformer | reported in both directions |
| H6.2 | Streaming inference cost per token is nearly independent of context length | CM (M7) | ms per token at N=65536 vs N=16, streaming mode | at most 2x (the inference form of H3.3) | the measured slope is stated; the near-constant claim is withdrawn |
| H6.3 | Maximum-batch throughput on the V100 is reported for the TM, the transformer and BERT-base on the same positions | E10 (M7, M9) | tokens per second | no threshold | reported in both directions |
| H6.4 | Incremental re-evaluation lowers generation latency per token | T2.3 (M8) | ms per generated token with and without it, same prompts | lower with it | reported; the decoder without it stays the reference |
| H6.5 | Joules per token at batch 1 on the V100 are below the from-scratch transformer's | E10 (M9) | joules per token, NVML, idle subtracted | TM below | reported in both directions |

Also fixed here for the scaling points (docs/04-implementation-plan.md, B1 and B3): every s1 and s2 point trains on the same number of examples as the anchor run; that number and the epoch budget are written into amendment A2 with the tiny-LM log path before the first big run starts.

### A2 (2026-09-22): epoch budget and the example count of the scaling points
Reason: A1 and docs/05 defer the epoch budget to the tiny-LM outcome. The tiny LM under H1.4 trains
4 epochs over its 4,999,168-token slice (masked mode, 15% of positions per epoch), the value fixed in
`experiments/configs/tiny_lm.json` before the run. Seed 1 gives top-1 0.1841, 0.1848, 0.1868, 0.1891
after epochs 1 to 4 [measured: `results/20260922T190301Z_tiny_lm/`]: no plateau within the budget,
the curve is still rising by about 0.2 points per epoch. Consequences, fixed here: (1) H1.4 is judged
on the epoch-4 value of each seed, as run; (2) the anchor run (CP_anchor, docs/05 CP) and every s1 and
s2 point train on the same number of examples, 4 epochs of their own corpus, and report the per-epoch
validation curve so a plateau or its absence is visible; (3) if the anchor run is still rising at its
last epoch, the report says so and no "converged" wording is used. Nothing above is changed.

### A3 (2026-09-24): the Count-Vote Tsetlin Machine (CVTM), hypothesis family H8
Reason: the signed-vote engine is far below KN-5 (validation 2.1749 bits per byte, test 2.2193, KN-5
1.3501 and 1.3719 [measured: `results/20260923T042350Z_CP_anchor/`]), and the evaluation reported in
`docs/results.md` found that count-derived votes repair the TM's estimator
while its CKM read-out fails. On 2026-09-24 it was decided that the necessary change be made
(docs/12 F19). The model is specified in docs/11 section 11. Nothing above is changed; H1.1 stays NOT
MET on seed 1 and is not re-judged by the CVTM, which is a different model.

Disclosure of what was seen before these thresholds were set: the exploratory runs
`results/20260924T001024Z_ckm_probe_masked_exploratory/`, `results/20260924T012544Z_ckm_probe_causal_exploratory/`,
`results/20260924T022628Z_ckm_probe_causal_controls_exploratory/`,
`results/20260924T032913Z_ckm_probe_causal_controls_20M_exploratory/`,
`results/20260924T035633Z_ckm_probe_causal_scale_compare_exploratory/`,
`results/20260924T021631Z_tm_levers_feasibility_causal_exploratory/`,
`results/20260924T022949Z_tm_levers_trigger_control_causal_exploratory/` (about 180 rows, all on the
validation split, counts from at most a 20M-token training prefix, the masked seed-1 bank and the CP
bank). Confirmatory evidence for H8 is therefore only: fits on the full training split, the unseen
masked banks of seeds 2 to 5, new banks for H8.10, and the test split accessed once per model after
the model is frozen. The CVTM-core is deterministic given the data (no seed); uncertainty is reported
as a document-block bootstrap (validation documents, 2,000 resamples), labelled derived.

The prediction column is the probability that the threshold is met [unverified
judgement, written before any H8 run]. A miss is reported as a negative result with its log path.

| Id | Hypothesis | Measurement | Threshold (target) | Prediction |
|---|---|---|---|---|
| H8.1 | Count-derived votes repair the engine's own estimator (T1) | the section 11.5 stand-alone read-out vs the same bank's signed votes: masked tiny-LM banks of seeds 2 to 5 (`results/20260923T125642Z_H2_row_d/`), 56,210 frozen targets; the CP anchor bank, causal, counts from the first 20M training tokens, full validation split | masked top-1 at least 0.05 higher on each of the 4 banks; causal validation bits per byte at least 0.15 lower | 0.9 |
| H8.2 | Kneser-Ney is exactly a clause-vote machine | the suffix clauses of section 11.2 with float64 votes vs `KNModel.prob`, every validation position; the Q16 integer form | max absolute difference of ln P at most 1e-9; the integer form changes validation bits per byte by at most 0.001 | 0.95 |
| H8.3 | P1 headline: the CVTM-core beats KN-5 on identical tokens | test bits per byte, once, after freezing; KN-5 test 1.3719 | at most 1.3219 (KN-5 minus 0.05) | 0.9 |
| H8.4 | The CVTM-core beats the classical cache LM | test bits per byte vs KN-5 plus a document unigram cache fitted by the same protocol (section 11.6) | lower | 0.95 |
| H8.5 | Energy: at equal training joules the CVTM-core beats the from-scratch transformer | NVML joules of the CVTM-core fit (GPU steps, idle subtracted, exclusive lock; host CPU seconds reported, CPU energy is not readable in this container); the in-house L6 W256 transformer (settings of `results/20260922T172902Z_M2_baselines_transformer/`) trained with a time cap that matches those joules (cap = joules / 259.2 W, the measured mean power of its 600 s run [derived: 155,519.8 J / 600 s]), rerun once if its joules differ by more than 10 percent | CVTM-core validation bits per byte lower; the joule ratio to the 600 s transformer is reported without a threshold | 0.9 |
| H8.6 | Placement against the 600 s transformer (1.2341 validation, 155.5 kJ) | validation bits per byte | no threshold, reported both ways (predictions: within 0.05 bits per byte 0.7; below it 0.2) | as stated |
| H8.7 | Context at constant cost | CVTM-core validation bits per byte with document window N in {0, 16, 64, 256, 1024, 4096, 16384, 65536, whole document}; ns per token of the scoring pass, single thread, median of 5 repeats (contended host, labelled) | (a) bits per byte non-increasing in N within 0.001; (b) per-token perplexity at N = 4096 at least 5 percent below N = 0; (c) ns per token at N = 65536 at most 1.5 times N = 16 | (a) 0.9, (b) 0.95, (c) 0.85 |
| H8.8 | Scaling with data and bit-exact sharding | (a) CVTM-core validation bits per byte at training prefixes of about 5M, 20M, 67M tokens and the full split (whole documents); (b) counting with S = 4 shards vs unsharded; (c) end-to-end fit throughput on the V100 | (a) strictly decreasing; (b) bit-identical counts and votes; (c) at least 1M training tokens per second | (a) 0.97, (b) 0.99, (c) 0.85 |
| H8.9 | Memory: the model can be pruned at little cost | CVTM-core with n-gram types of count 1 removed at orders 3 to 5, then count at most 2; model bytes vs validation bits per byte | a variant at no more than 30 percent of the unpruned bytes loses at most 0.02 bits per byte | 0.7 |
| H8.10 | Tsetlin LEARNING adds accuracy (T2) | section 11.5 residual read-out of learned banks on top of the CVTM-core, banks trained on the SAME tokens as the counts (5M and 20M prefixes, 65,536 clauses, causal), controls C0 (a young bank of equal size) and G+ (enumerated class, skip and trigger clauses of equal count); validation | gain at least 0.01 bits per byte over the CVTM-core AND at least 0.005 above the larger of the C0 and G+ gains | 0.2 |
| H8.11 | Masked CVTM (Gate 1 amendment row, docs/11 section 11.8) | top-1 on the V 8000 frozen validation targets; comparators left bigram true context 0.1687, CBOW-kNN 0.0936, KN-5 both sides 0.4419 [measured: `results/20260923T203943Z_E3_v8k_slice_exploratory/` comparator rows] | above the left bigram and CBOW-kNN; KN-5 both sides reported (prediction of beating it 0.7) | 0.95 |

What is NOT claimed under A3: the CVTM-core contains KN-5 as a component, so "beats KN-5" (H8.3) says
nothing about Tsetlin learning; its accuracy comes from counting and in-context copying expressed as
clause votes. Only H8.1 (the TM's own bank, repaired) and H8.10 (Tsetlin learning over the core)
credit the Tsetlin machine with accuracy. No claim is made that the CVTM matches a transformer trained
beyond 600 s.

### A4 (2026-09-24): TA-learned residual clauses, the transformer frontier, white-box properties, the kinship world, and PG-19 (families H9, H10, H11)

**Reason.** A3 credits the CVTM-core's accuracy to counting. Tsetlin learning is credited only through H8.1 (MET: `results/20260924T054640Z_cvtm_H8_1_masked_seed2/` to `..._seed5/`, `results/20260924T062704Z_cvtm_H8_1_causal_cp20M/`) and H8.10. This amendment registers:
- binary TA-learned clauses aimed at the core's errors;
- a tuned neural comparator;
- white-box properties;
- the reasoning task through which Granmo defines C1 to C6 (README of `cair/HierarchicalTsetlinMachine`, 2026-02-06).

Nothing above is changed.

**Disclosure.** Seen before these thresholds were set:
- `results/20260924T054225Z_cvtm_early_fulldata_exploratory/` and `results/20260924T060608Z_cvtm_early_fulldata_mixcheck_exploratory/`;
- `results/20260924T055525Z_cvtm_early_logical_attention_pilot_exploratory/`, `results/20260924T061202Z_cvtm_early_logical_attention_exploratory/` (exit 143) and `results/20260924T062408Z_cvtm_early_logical_attention_exploratory/`;
- the H8.1 runs;
- scratch analyses that are not citable: BLiMP scored for KN-2, 3 and 5 and for the 600 s transformer; a complementarity decomposition; a toy simulation of the vote equilibrium; an inspection of the CLUTRR training files.

No test split was accessed.

**What counts as confirmatory evidence:**
- seeds 1, 2 and 3 (section 3);
- the full training split unless stated otherwise;
- selection only on held-out training documents or on validation;
- the test split once per frozen model.

**Held-out training documents H'.** Documents whose splitmix64 residue (docs/11 section 11.6) is in {0, 1, 2}. Residue 0 fits the core's EM, residue 1 trains the T2 clauses, and residue 2 is used for selection, temperature and mixture weights. The T2 inputs use a KN-5 fitted without H'.

**Controls in every H9 row:**
- LR on the same literals;
- LR on all pairwise literal products;
- histogram gradient-boosted trees (scikit-learn);
- C0, a TM of equal size trained on 1 percent of the pairs.

Each control gets the same number of configurations evaluated on residue 2 as the TM.

**New data.** The hashes go into a new write-once file `docs/notes/preregistration-data-hashes-A4.json`:
- BLiMP (67,000 pairs);
- CLUTRR `gen_train23_test2to10` (generator licence CC BY-NC 4.0; the dataset card says unknown);
- a seeded synthetic kinship generator;
- PG-19 (Apache-2.0).

**Seeds.** Exploratory pilots use seeds of 100 or more and are never reported as confirmatory.

| Id | Hypothesis | Measurement | Threshold (target) | Prediction |
|---|---|---|---|---|
| H9.1 | The TA-learned copy selector adds accuracy over the CVTM-core | Binary TM (Type I and II feedback) over context, candidate and reflective literals. Candidates: earlier document tokens, cap 1024. Residual P proportional to P_core·exp(beta_b·r) with an exact sparse normaliser. Full-data base; validation; 3 seeds | Validation bpb at least 0.010 below the CVTM-core on every seed, document-bootstrap CI excluding 0 | 0.5 |
| H9.2 | The copy selector's TM is essential | Same pairs and literals; best of the named controls | TM minus the best control at most 0.000 bpb is reported as "matches"; at most -0.002 with the CI excluding 0 as "beats"; otherwise "trails" | matches 0.4; beats 0.15 |
| H9.3 | CR-TM: TA learning aimed at the core's errors | Binary TM. Candidates: top-64 of P_core plus the copy candidates. Literals add candidate clusters, eq_k (the candidate's cluster equals the cluster at offset -k) and bag membership. K = 4 negatives drawn from P_core. 20M, then the full split; 3 seeds | Validation bpb at least 0.010 below the best of {CVTM-core, CVTM-core plus H9.1} AND at least 0.005 below the best control and C0 | 0.15 |
| H9.4 | Syntax (C1): BLiMP | Sentence log-probabilities with identical preprocessing for every model. Half of each paradigm's pairs are dev; the other half is reported once | CVTM plus CR-TM at least CVTM-core plus 3.0 points AND at least the best control plus 1.5; paired bootstrap over pairs, CI excluding 0 | 0.2 |
| H9.5 | Faithfulness of H9.4 | Remove the clauses that contain relational literals | Removes at least 50% of the H9.4 gain | 0.5, given H9.4 |
| H9.6 | Vote equilibrium: S = T·tanh(0.5·ln(p/b)), b = (1-p)Q/(V-1) | CP bank, 200,000 validation positions, the 50 most probable KN-5 classes per position; bins of x = ln(p_KN / b); mean S/T per bin against tanh(x/2) | Weighted R² at least 0.8, AND Spearman(S, ln p) at most 0.2 among classes with p > 20b | 0.55 |
| H9.7 | Transformer frontier and crossover E*(D) | Family {L6 W256, L6 W384, L8 W512, L12 W768}, fp16 AMP, context 1024, learning rate chosen at the 150 s cap on validation; caps {150, 600, 2400, 9600} s; NVML joules with idle subtracted, exclusive lock. CVTM-core fit joules at the H8.8 prefixes. E*(D) = the smallest transformer joules that beat the CVTM-core fitted on D tokens | No threshold; reported both ways. Predictions: the tuned 600 s model beats 1.2409 (0.6); E*(full) is at least 5 times the CVTM-core fit joules (0.6) | as stated |
| H9.8 | Self-assessment at LM level (C2) | CVTM-core, held-out validation half | ECE (15 bins, top-1 confidence) at most 0.05; split-conformal coverage at least 0.90 at alpha 0.1, with set size reported; support-signal AUROC for top-1 correctness at least 0.80, with the tuned transformer's max-softmax AUROC reported | 0.7; 0.85; 0.5 |
| H9.9 | Exact deletion and insertion (C4, LM level) | 100 random training documents deleted and 100 inserted by count deltas, stable document IDs, compared with a refit | Array hashes equal in 100 of 100 cases each; median latency reported against the refit. A property check, positive by construction | 0.9 |
| H9.10 | Receipts for every CVTM token (C3) | Q16 receipts; standalone checker in a separate process | 100% of the 282,587 validation tokens pass; 100% of 10,000 tampered receipts fail | 0.9 |
| H9.11 | Provenance, "as documented" (C3) | Suffix array over the training tokens | 100% agreement with brute force on 1,000 random tokens | 0.9 |
| H10.1 | C5 on gold edges | Composer TM trained on k = 2 only; forward chaining (generic, hand-written, declared); CLUTRR | Pooled accuracy over k = 6 to 10 at least 0.90; per-k curve; count table and R5/CTP rows as placement | 0.75 |
| H10.2 | C5 plus C1 from text | TA-trained reader plus composer | Pooled accuracy over k = 6 to 10 at least 0.80 AND above an equal-cost from-scratch transformer, CI excluding 0; reader F1 reported against the transformer | 0.35 |
| H10.3 | Rule recovery | Learned composition clauses against the ground-truth table | At least 95% of the pairs seen in training recovered exactly; held-out combinations reported | 0.8 |
| H10.4 | C2 abstention | Synthetic set, 30% of queries made underivable by deleting one supporting sentence; text input | AUROC at least 0.95; the transformer's max-softmax AUROC reported | 0.6 |
| H10.5 | C3 proofs | Every answer | 100% carry a checker-verified chain; gold-path exact match at least 0.90 among correct CLUTRR answers (`proof_state`) | 0.7 |
| H10.6 | C4 one-shot insertion | One new sentence inserted; the queries it entails | At least 0.90 correct with no retraining; earlier queries lose at most 1 point; transformer fine-tuned at equal compute reported | 0.7 |
| H10.7 | C6 | (a) A controller on the C2 margin chooses a direct clause answer or chaining. (b) TA compilation over a recurring query stream | (a) Accuracy within the 3-seed range of chaining-only at no more than 50% of its clause evaluations. (b) Median latency at least 2 times lower, with accuracy within the seed range | 0.45 |
| H10.8 | C1 abstraction (Granmo's talk, pages 16 and 17) | Hierarchical TM against a flat TM, LR, an MLP and a count table, on held-out relation combinations whose gender-swapped siblings were seen; 5 seeds | Hierarchical at least 0.90 AND at least flat plus 0.20 | 0.3 |
| H10.9 | Slide 12 end to end | Derived facts written into the CVTM document memory; generated answer sentence | Exact match at least 0.80 at k = 4 to 10, from text | 0.4, given H10.2 |
| H11.1 | Data scaling on PG-19 | CVTM-core at 20M, 100M, 500M and the full split; BPE retrained on PG-19 train; near-duplicate audit | Validation bpb strictly decreasing; de-overlapped numbers reported | 0.7 that the full point runs; 0.97 monotone if it does |
| H11.2 | Sharded counting at scale | S in {1, 4, 16} at 100M tokens | Bit-identical | 0.95 |
| H11.3 | Whole-book context at constant cost | ns per token for the whole book against N = 16 | At most 1.5 times | 0.85 |

**Gates (decision rules fixed now, applied in this order; a gate never changes a threshold above).**
- G1 frame (after the tuned-transformer pilot of H9.7 at 150 s and 600 s): if the tuned 600 s transformer reaches at most 1.2209 validation bits per byte (CVTM-core minus 0.02), the LM result is framed only as the energy frontier E*(D) plus the white-box properties, with no parity sentence; otherwise a parity sentence is allowed against the TUNED 600 s comparator only.
- G1 T2 (after H9.1 at full data, 3 seeds): a gain of at least 0.010 with the CI excluding 0 on every seed opens the CR-TM build (H9.3); a gain between 0.005 and 0.010 keeps the copy selector as a row and runs H9.3 only as a 5M pilot; below 0.005 no T2 engine work is done and T2 is reported negative after H8.10.
- G1 T1 (after H9.6): a weighted R-squared of at least 0.8 opens a same-data bank trained on the positions the core predicts poorly; the engine change toward NCE-style feedback follows only if that bank beats a uniform-negative bank by at least 0.005 bits per byte.
- G1 K (after the gold-graph pilot of H10.1 at seeds of 100 or more): pooled accuracy at k = 10 of at least 0.95 opens the text reader (H10.2); 0.6 to 0.95 allows one day on the composition order; below 0.6 only the gold-edge demonstrations of C2, C3 and C4 are run and C5 is reported negative.
- G2 (day 4, if G1 T2 opened H9.3): at 20M tokens, a gain of at least 0.005 over the CVTM-core plus the copy selector and above the best control, or H9.3 stops and is reported.
- G3 (day 5, if G1 K opened H10.2): H10.2 of at least 0.80 on validation, or a text-level negative is reported.

**What is NOT claimed under A4:**
- T1's accuracy is counting.
- T3's chaining is hand-written.
- H9.9 to H9.11 are property checks.
- Gold-edge results are not text understanding.
- No "LLM" wording, and no "resolved" wording for C1 to C6.

### A5 (2026-09-24): continue/pivot decision rule, early verdict, and exploratory pilots P1 and P2

**Reason.** On 2026-09-24 the question was raised whether stage 2 delivers enough evidence to decide between continuing the current approach and switching. A4 fixes one gate per strand, but not what the combined outcomes mean. Its T2 verdict can arrive only after H8.10 and G2 (days 2 to 4). This amendment fixes the following before any input below has been seen:
- (a) the ambiguities A4 left open (D0.1 to D0.3);
- (b) an early, provisional verdict built from B1 and a pilot on the existing full-data cache (P1);
- (c) a count-only pilot for the learned-representation family (P2);
- (d) the decision table.

No threshold of H1 to H11 is changed, no A4 gate is removed, and nothing registered is cut (protocol rule 17).

**Disclosure.** Seen before this amendment:
- everything disclosed in A3 and A4;
- `results/20260924T085728Z_cvtm_cache_build/` (the frozen CVTM-core, validation 1.240880 bits per byte);
- `results/20260924T082109Z_cvtm_cache_build/` (exit 1, a logging error after the fit);
- scratch analyses, not citable, all on validation or on H' of the cache:
  - the gold's rank in P_core;
  - a cross-fitted mixture of the CVTM-core and the 600 s transformer, broken down by rank, frequency, history and k\*;
  - two extrapolations and one cross-fitted-mixture projection of the 5M copy selector onto the full-data core, from `results/20260924T062408Z_cvtm_early_logical_attention_exploratory/per_token_bits.npz`.

Not seen: any output of the tuned-transformer pilot, H8.5, H9.1, H9.2, C0, the kinship pilot, H8.8a, H9.6 or H8.10. Smoke runs of lane GX started at 10:15 UTC; any smoke output read before this amendment is saved is listed here: none (no output of any stage-2 lane was read before this amendment was saved at 10:24 UTC; only process names and run-directory names were seen). Any input read before this amendment was saved makes the rows it feeds "not pre-registered" in the report.

**Definitions.**
- **D0.1 (H9.1 band; clarifies G1 T2).**
  - Band A: every seed gains at least 0.010 bits per byte, each with its document-bootstrap CI excluding 0.
  - Band B: every seed gains at least 0.005 with its CI excluding 0, and the result is not band A.
  - Band C: otherwise.
  - The band is set by the smallest per-seed gain, and the seed mean is reported alongside it. A missing seed is rerun; it is never skipped.
- **D0.2 (H9.2 verdict).**
  - The best control is chosen on residue 2 of H', never on validation.
  - The verdict uses the 3-seed mean of (TM minus best control), with a paired document bootstrap (validation documents, 2,000 resamples).
  - A control whose fit stops at its iteration cap is refitted once with the cap raised four-fold before the verdict. If it still stops at the cap, that fact is reported and "beats" is not claimed.
  - TM minus C0 is reported as its own row.
- **D0.3 (learning credit; a label used only by this rule).**
  - The copy selector has credit when it is in band A and TM minus C0 is at most -0.005 bits per byte with the CI excluding 0.
  - H8.10, H9.3 and the G1 T1 bank have credit when they meet their registered thresholds (the H8.10 thresholds for the G1 T1 bank).
  - P1 has credit when it is GO.
- **P1 (exploratory; seed 100 first, seeds 101 and 102 only in the extension): CR-TM on the frozen cache.**
  - **Base:** the cache of `results/20260924T085728Z_cvtm_cache_build/`, i.e. P_core top-64 on H', computed with the KN-5 fitted without H'.
  - **Training:** 250,000 positions of residue 1, each with the gold and K = 4 negatives drawn from P_core top-64.
  - **Selection:** residue 2a, half of residue 2's 261 documents, split by document.
  - **Judgement:** residue 2b (the other half), document bootstrap, 2,000 resamples. Validation and test are not used.
  - **Model:** a binary TM with about 900 literals: clusters 16 and 243 at offsets -1 to -8, the candidate's clusters and flags, eq_k, cluster-bag membership, reflective buckets, and classical copy features. 4,000 clauses; the candidate set is the top-64; exact normaliser; beta per bucket fitted on 2a.
  - **Controls** (same literals and pairs, the same number of configurations on 2a): LR, LR on pairwise products, histogram gradient-boosted trees, and C0 (a TM of equal size trained on 1 percent of the pairs).
  - **Ablations:** no relational literals; context-only literals plus the candidate id (the NCE-bank objective without the engine change).
  - **Arm (ii):** the same TM and controls scored on the BLiMP dev half (the H9.4 split; data hashed per A4 before use).
  - **GO:** either of
    - (i) residue-2b bits per byte at least 0.012 below the core with the CI excluding 0, AND TM minus the best of {LR, pairwise LR, GBDT} at most -0.003 with the CI excluding 0, AND TM minus C0 at most -0.005;
    - (ii) the BLiMP dev half at least 2.0 points above the core (paired bootstrap over pairs, CI excluding 0) AND at least 1.0 above the best control.
  - **NO-GO:** arm (i) gains less than 0.005 or trails the best control, AND arm (ii) is below 1.0 point over the core or trails the best control. An arm that was not run is named as untested.
  - **Grey:** anything else. One extension is allowed (copy candidates added, 8,000 clauses, seeds 101 and 102) with the same thresholds. Still grey after it counts as NO-GO for N\*-early.
  - **Role by band:**
    - band A: P1 is the first step of H9.3;
    - band B: P1 serves as the G1 T2 pilot in place of a 5M refit. This substitution needs a recorded decision; without it, the 5M pilot also runs as registered;
    - band C: a GO does not open H9.3, because that would change G1 T2. It is recorded as an open decision.
  - P1 supports no claim.
  - **Prediction:** GO 0.2.
- **P2 (exploratory; counting only, no TM): the ceiling of class and skip templates at full data.**
  - Templates: c2_1, c2_21, t1_c2_2, t1_c3_2, c3_321, t21_c2_3, t32_skip and t321_c2_4, counted over the training split without H'.
  - Class maps: the repository's exchange clusters (243 and 2,497), PPMI-SVD k-means (256 and 2,048), and a random partition of equal size.
  - Mixed into the frozen core with k\*-by-count-bucket weights fitted on residue 2a, judged on residue 2b.
  - G_best is the best real map's gain; R is G_best minus the random map's gain.
  - NO-GO for this family on bits per byte: G_best below 0.005 or R below 0.003.
  - GO to a TM concept-map step: G_best at least 0.010 with the CI excluding 0.
  - In between: TM concepts enter only inside H9.3, as cluster literals.
  - H8.10, including its G+ arm, is unchanged.
  - **Prediction:** NO-GO 0.8.
- **N\*-early.** Holds when the copy selector has no credit AND P1 is NO-GO.
  - It is decided as soon as B1 and P1 are both in.
  - Effect: RESCOPE is provisional. No new engineering aimed at Tsetlin-learned LM accuracy starts: no H9.3 beyond what G1 T2 and G2 require, and no NCE engine change.
  - Registered rows still run.
  - Any later credit from H8.10, G2 or the G1 T1 bank revokes it.
  - It is an allocation verdict, not a reported result.
- **N\* (final).** No component has credit among the copy selector, P1, H8.10, H9.3 (at G2 if opened) and the G1 T1 bank (if opened). A component that was not run counts as having no credit and is named as untested.
- **T3 credit.** H10.2 or H10.8 is met. G1 K and H10.1 alone give none, because on gold edges the count table is the placement row and the chaining is hand-written (A4).

**Decision classes.**
- **CONT:** continue as registered.
- **CONT+L:** CONT plus investment in Tsetlin learning: the H9.3 build through G2, and the G1 T1 engine path under its own rule.
- **RESCOPE:**
  - LM accuracy is credited to counting and copying in clause form;
  - the TM is credited with the clause form (integer votes, receipts, exact insertion and deletion, constant-cost context, calibration), with H8.1, and with the reasoning tier;
  - no further engine work aims at Tsetlin-learned LM accuracy, and the negative result is reported with its runs.
- **SWITCH:** a change of the thesis question, chosen by a recorded decision and never by this rule. Named options:
  - SW-H: the CVTM-core as the exact, editable memory of a small neural LM;
  - SW-G: a gradient-trained logic-gate network (Petersen et al., NeurIPS 2022).

  A SWITCH needs its own pre-registered pilot before it replaces anything.

The prediction column is the probability that the row is reached [unverified judgement, written before any input was seen].

| Id | Condition | Decision | Claim allowed | Prediction |
|---|---|---|---|---|
| D1 | G1 frame met (tuned 600 s at most 1.2209) | framing only | E\*(D) plus the white-box properties; no parity sentence | 0.5 |
| D2 | G1 frame not met | framing only | "within X of a tuned 600 s transformer on one V100", against the tuned comparator only | 0.5 |
| D3 | H8.5 met, or missed | framing only | met: lower validation bits per byte at equal GPU training joules; missed: reported, and the efficiency claim is withdrawn | 0.9 met |
| D4 | H8.8a not strictly decreasing | confirmatory LM rows paused; diagnose; no switch | none until the cause is found | 0.05 |
| D5 | band A, credit, H9.2 "beats" or "matches" | CONT+L | "the Tsetlin-learned copy selector adds X (at least 0.010) over the CVTM-core on 3 seeds and beats/matches the named controls" | 0.10 |
| D6 | band A, credit, "trails" | CONT+L bounded; no GPU engine mode before G2 | "adds X; the best non-TM learner on the same literals adds Y more" | 0.10 |
| D7 | band A, no credit | CONT; H9.3 opens under G1 T2 | "the gain comes from the literals and the read-out, not from Tsetlin learning beyond 1 percent of the pairs" | 0.15 |
| D8 | band B | CONT; the selector stays as a row; P1 | "adds 0.005 to 0.010, below the registered 0.010" | 0.35 |
| D9 | band C | provisional RESCOPE for the copy selector; P1 | "the exploratory 5M gain of 0.0412 does not reach 0.005 at full data" | 0.30 |
| EV1 | the copy selector or P1 has credit | CONT+L as in D5 to D8 | none from P1 itself | 0.35 |
| EV2 | N\*-early | provisional RESCOPE, as defined above | none until N\* is final | 0.60 |
| D10 | H9.6 met | G1 T1 bank test as registered (P1's context-only ablation is its early reading) | a mechanism claim, not an accuracy claim | 0.55 |
| D11 | H9.6 not met | no NCE engine change | "the saturation account is not supported" | 0.45 |
| D12 | G1 K at any level | as A4; the LM approach is unchanged | TM-learning credit in T3 only through H10.2 or H10.8 | as A4 |
| D13 | N\* and T3 credit | RESCOPE | LM credited to counting and copying in clause form; H8.1; a controlled negative for Tsetlin-learned LM accuracy, naming the tests; Tsetlin learning shown on reasoning | 0.20 |
| D14 | N\* and no T3 credit | RESCOPE is the report; SWITCH becomes eligible for a recorded decision | "in N registered tests Tsetlin learning added no LM accuracy beyond non-TM learners or C0; its only accuracy result is H8.1, at KN-2 level" | 0.30 |
| D15 | H8.3 missed on test, or H9.9 or H9.10 missed after a fix | stop and diagnose (protocol rule 18); no switch before the cause is known | as registered | 0.05 |

**What is NOT claimed under A5:**
- A transformer beating the CVTM (G1 frame, H9.7, TB) is framing only and never a trigger to switch.
- N\*-early, N\*, RESCOPE and SWITCH are statements about Tsetlin learning on WikiText-103 (135M training tokens, V 8000) under the named tests, not about Tsetlin machines in general.
- P1 and P2 support no reported claim.
- Whichever row is reached, the CVTM results of A3 are reported with their credit to counting unchanged.

### A6 (2026-09-24): scaling ladders of the Tsetlin-form LM (data, learned generalization, depth, efficiency) and the bounded distillation pilot (families H12 to H16)

**Reason.** The goal of 2026-09-24 (docs/12 F21 (b), F22, F23): scale the Tsetlin machine toward transformer-LM quality while keeping its advantages, teacher-free on the main path, with distillation only as one bounded and separately labelled pilot. This amendment registers how far each route goes, earliest decisive test first. Nothing above is changed: no threshold of H1 to H11, no gate of A4, no definition or decision row of A5. The A5 N\*-early rule stays in force: H13.1 and H14 are new engineering aimed at Tsetlin-learned LM accuracy, so they start only before N\*-early is decided or on an explicit recorded decision. The self-supervised family of docs/12 F23 is registered separately as amendment A7 (family H17), after its exploratory early tests report.

**Disclosure.** Seen before this amendment was saved:
- everything disclosed in A3, A4 and A5;
- A3 outputs read after A5 was frozen: `results/20260924T102203Z_cvtm_a3_H8_2/`, `results/20260924T102454Z_cvtm_a3_H8_7/`, `results/20260924T110429Z_cvtm_a3_H8_8b/`, `results/20260924T110429Z_cvtm_H8_11_masked/`, and the partial `results/20260924T104118Z_cvtm_H8_5_fit_energy/` (exit 143, 3 of 5 repeats, whole-fit joules read);
- logs of runs ended by SIGTERM: `results/20260924T112441Z_cvtm_H8_5_fit_energy/`, `results/20260924T112548Z_cvtm_a3_H8_8a/`, and `results/20260924T113130Z_cvtm_pilot_p2_exploratory/` (it printed only the core's bits per byte on residues 2a and 2b, 1.25808 and 1.25629);
- the last log lines of `results/20260924T110143Z_cvtm_copy_selector_pairs/`;
- scratch analyses, not citable:
  - (i) on validation: the gap anatomy of the CVTM-core against the 600 s transformer (cross-fitted mixture saving 0.140 bits per byte with a context-only gate), and hand-written state clauses, recency caches and a projection of the 5M copy selector (together 0.0162);
  - (ii) KN-5 data-value probes at 0.5M and 5M added tokens of Wikipedia 20231101 and FineWeb-edu, reformatted and raw;
  - (iii) on H' residue 2 including 2b: eight context-similarity retrieval heads (0.0015), and document caches keyed by the core's class posterior (0.0034 at 16 classes, 0.0055 at 16 plus 256) with oracle bounds;
  - (iv) energy-per-token literature and arithmetic.

Not seen: any output of GX, H9.1, H9.2, H9.6, H8.10, P1 (its unlogged scratch files were not opened) or the kinship pilot. Read besides the A3 outputs listed above: the logged prefix rows of the killed H8.8a run (`results/20260924T112548Z_cvtm_a3_H8_8a/`: CVTM-core 1.3812 at 20M and 1.2914 at 67M training tokens) and the first result line of H8.11; no output was read of GX, H9.1, H9.2, H9.6, H8.10, P1 or the kinship pilot, and of P2 only the log listed above. Any input read before saving makes the rows it feeds "not pre-registered".

**Definitions.**
- **H'' (fresh held-out documents).** Training documents whose splitmix64 residue (docs/11 section 11.6) is in {3, 4, 5}.
  - A second CVTM-core is fitted by the frozen protocol on the training documents outside H' and H'' (EM on residue 0). Its cross-fitted outputs (gold probability, top-64, reflective literals) on residues 0 to 5 are cached once, with hashes.
  - Roles: residues 1 and 3 train; residue 2 holds auxiliary counts; residue 4 selects; residue 5 judges (document bootstrap, 2,000 resamples).
  - The frozen A3 core and every A3 to A5 row are unchanged.
- **C-rule.** The report may say "within X bits per byte of transformer T" only if T was trained on the same token stream, and T's parameter count, training tokens, GPU hours and joules are stated. "Comparable" requires X at most 0.02.
  - Published models (Merity et al. 2018, Transformer-XL, kNN-LM, GPT-2) appear only as derived placements, labelled not like for like.
  - "LLM" wording stays forbidden (A4).
- **DC (decontamination) for any added corpus.**
  - Drop articles whose normalised title matches a WikiText-103 train, validation or test title.
  - Drop documents containing any BPE 13-gram of validation or test that is not in WikiText-103 train.
  - The test 13-gram set is built once, as one counted test access with purpose "decontamination", before the first rung's data is frozen.
  - Infini-gram's 80 percent rule is reported only as a sensitivity row.
- **E-rule.**
  - GPU joules: NVML at 50 Hz under the exclusive lock, raw and idle-subtracted.
  - CPU joules: derived as process CPU seconds times {4.27, 6.0, 8.54} W, and never called measured (RAPL is not readable in this container). Both sides pay for their host CPU.
  - Resident model bytes are stated beside every ratio: CVTM-core 6,541,415,381 bytes; L6 W256 checkpoint 27,440,151 bytes [measured: file sizes in `results/20260924T085728Z_cvtm_cache_build/` and `results/20260922T172902Z_M2_baselines_transformer/`].

**Family H12: data scale.** Counting in clause form; no Tsetlin-learning credit.
- **Rungs.**
  - R0: WikiText-103 train.
  - ES1: about 150M decontaminated, reformatted tokens from three seeded shards of `wikimedia/wikipedia` 20231101.en.
  - R1: plus Good and Featured articles absent from WikiText-103.
  - R2: about 0.5B tokens.
  - R3: all cleaned Wikipedia.
  - R4: plus FineWeb-edu.
- **Method.** Reformatting uses sacremoses with WikiText-103's number splits and heading form. New dependencies get a docs/12 row first. Metric: validation bits per byte with the fixed estimator of docs/11 section 11 and the same byte accounting. Rows are deterministic, with a document bootstrap.
- **Long-match clauses** (only if H12.2 is met):
  - Read-out: successor counts of the longest training match of length 6 or more, from a suffix array.
  - Residual vote: r(w) = ln(lambda_S + sum_i lambda_i P_i(w) / P_KN(w)).
  - Weights: MAP-EM over buckets (match length 5 to 7, 8 to 11, 12 to 15, 16 to 31, 32 or more; count band; k\*; m\*).
  - The suffix array used for the EM excludes the EM documents. The clauses are tested against brute force before any rung number is read.
- **Gates.**
  - ES1 per H12.1.
  - R2 to R3 only if R2 is at most 1.19.
  - R3 to R4 only if eta of FineWeb-edu at 60M is at least 0.10.

| Id | Hypothesis | Measurement | Threshold (target) | Prediction |
|---|---|---|---|---|
| H12.1 | ES1 decides whether more Wikipedia moves WikiText-103 validation (a decision rule, not a claim) | KN-5 on W60, W60+X60, W120 and X60; eta_60 = 2^((b(W60) - b(W60+X60)) / (b(W60) - b(W120))) - 1; KN-5 and CVTM-core on W135+X135 | GO to R2 and R3: eta_60 at least 0.25 AND KN-5 at most 1.3321 AND CVTM-core at most 1.2259, with the gain's CI excluding 0. NO-GO: eta_60 below 0.15 OR CVTM-core gain below 0.007; stop after R1 and report the negative. Otherwise run R1 and R2 and decide on R2 | GO 0.45; grey 0.40; NO-GO 0.15 |
| H12.2 | Higher order helps at 135M | KN-K for K in {6, 7, 8}, full split | KN-6 at least 0.006 below KN-5 (1.3501), CI excluding 0. If met, orders above 5 and long-match clauses enter every later rung; below 0.003, K = 5 is kept to R2 | 0.6 |
| H12.3 | Monotone data curve | CVTM validation bits per byte at every rung run | strictly decreasing | 0.95 given the rungs run |
| H12.4 | Contamination sentinel | share of validation positions whose longest match of at least 13 tokens occurs in the added data but not in WikiText-103 train; share at 8 or more against an equal-size WikiText-103 train slice | 0; at most 2 times the slice. A failing rung is not reported until fixed | 0.9 |
| H12.5 | Memory at scale | model bytes and mapped working set per rung, pruning by the H8.9 estimator | R3 runs under the 85 GiB guard with at most 30 GiB mapped; pruning loss reported | 0.7 given R3 |
| H12.6 | Equal-data transformer | H9.7 family on each rung's stream: R2 at 600, 2,400 and 9,600 s; R3 one run of 11 to 17 h, exclusive | none; reported both ways | transformer lower at R2 and above: 0.85 |
| H12.7 | Reach (prediction only) | best CVTM validation bits per byte on this ladder by 2026-10-15 | none | at most 1.20: 0.6; 1.15: 0.35; 1.10: 0.12; 1.00: 0.03 |

**Family H13: Tsetlin-learned generalization, additions only.** B1, P1, P2, H8.10 and H9.3 run as registered, and A5 governs investment.

| Id | Hypothesis | Measurement | Threshold (target) | Prediction |
|---|---|---|---|---|
| H13.1 | P3 (exploratory; seeds 100 to 102; judged on H'' residue 5): the TM discovers document-state logic from raw position-bound literals | A binary TM (Type I and II) over the literals "token t at offset -k" for t in { ( ) " = . , newline } and k in 1..32, plus sentence-start flags. Residual read-out of docs/11 section 11.5 (T2) on the cross-fitted core, restricted to the targets ) " . , : = newline. Controls on the same literals: the hand-written states (inside parentheses, inside quotes, heading line, at least 25 tokens since the last sentence end), LR, pairwise LR, GBDT and C0 | (a) For at least 3 of the 4 states, the F1 between "a clause voting positively for the state's target (), ", =, .) fires" and the state is at least 0.9. (b) The gain is at least 70 percent of the hand-written clauses' gain AND TM minus the best non-TM control is at most 0.000 | (a) 0.45; (b) 0.3 |

H13.1 is a rule-discovery test on natural text. Its pool is at most about 0.017 bits per byte [scratch], so it supports no LM-accuracy claim.

**Family H14: deep TM (PC-TM).** Exploratory pilot, seeds 100 to 102, on H''.
- **Model:**
  - base: the cross-fitted core;
  - concept tree: the exchange clusters (16, 243, 2,497), which are preprocessing and not Tsetlin-learned;
  - projection P_l: a coalesced multiclass TM over the children of the level-l node, with about 1,200 literals;
  - votes: v_ck = ln((O_ck + 0.5) / (E_ck + 0.5)), with O and E counted on residue 2;
  - retrieval: queried by Q_l.
- **Fixes relative to the scratch design:**
  - TM-1 is trained only on cross-fitted positions (residues 1 and 3);
  - the core's class masses come from the bias-class table, the followers of fired suffix clauses and the document candidates, so the cost does not depend on V (ns per token reported);
  - judgement is on residue 5 only.
- **Outcomes:**
  - GO-DEEP: G-proj and G-depth, and at least "matches";
  - GO-SHALLOW: G-proj only;
  - NO-GO: A2 minus A1 above -0.002, or the TM trails the best control by more than 0.002;
  - grey: one extension (8,000 clauses, seeds 103 and 104).

| Id | Hypothesis | Measurement | Threshold (target) | Prediction |
|---|---|---|---|---|
| H14.1 | Tsetlin-learned 16-way concept prediction adds over the core's own class posterior | A0: core. A1: core plus a concept cache queried by the core's class posterior. A2: core plus the TM-1 residual plus a concept cache queried by Q1. Controls through the identical read-out and cache: multinomial LR, pairwise LR, HistGBDT, class-KN-5 over k1 sequences, C0 | G-proj: A2 minus A1 at most -0.005, CI excluding 0, on every seed. G-TM: TM minus the best control at most 0.000 is "matches"; at most -0.002 with the CI excluding 0 is "beats"; TM minus C0 at most -0.003 | G-proj 0.2; "matches" given G-proj 0.5; "beats" 0.15 |
| H14.2 | Depth | A3 = A2 plus TM-2 over TM-1 outputs (about 3,300 literals). W = TM-1 with 2x clauses at equal TA exposures. R = TM-2 without TM-1 literals. A3-GBDT = GBDT on A3's literals | A3 minus W at most -0.003 AND A3 minus (A2 plus R) at most -0.003, both with the CI excluding 0, AND A3 minus A3-GBDT at most +0.001 | 0.25 given G-proj; 0.05 unconditional |
| H14.3 | Confirmatory; only after G-proj and a recorded decision to proceed | full-data PC-TM on validation, seeds 1 to 3 | at least 0.010 below the CVTM-core on every seed, CI excluding 0, AND G-TM at least "matches" | 0.06 unconditional |

**Family H15: efficiency of the CVTM rows.** A1's H6.1 to H6.5 are unchanged for the engine.
- **Rows:** the CVTM-core over the H8.7 windows; the H8.9 pruned variants; each TA add-on that A5 credits; the six M2 transformer checkpoints, the GX checkpoints and the H9.7 checkpoints.
- **Equal-quality rule:**
  - the transformer frontier is the lower envelope of (bits per byte, J per token) per platform and mode;
  - E_T(q\*) is interpolated log-linearly between the two members that bracket q\*;
  - the cheapest member at or below q\* is reported as the conservative number;
  - no extrapolation.
- **Platforms:** 1 CPU thread, 6 CPU threads, the V100.
- **Modes:** scoring; batch-1 generation (20 validation prompts, 32 tokens each); maximum batch.
- **Fairness to the transformer:** KV cache; CUDA graphs at batch 1; fp16 on the GPU; int8 dynamic quantisation on the CPU; best batch size.
- Every timed pass reproduces its model's bits per byte or reports the deviation.
- K1 is a C kernel of the core read-out (KenLM-style hash tables, Q16 votes).

| Id | Hypothesis | Measurement | Threshold (target) | Prediction |
|---|---|---|---|---|
| H15.1 | Batched, best platform each | ratio of energy per token, transformer to CVTM, at equal bits per byte | at least 10 | 0.75 (at least 100: 0.2; at least 1,000: 0.03) |
| H15.2 | CPU against CPU, batched | the same ratio, both on 6 threads (derived joules) | at least 100 | 0.8 (at least 1,000: 0.35) |
| H15.3 | Batch 1 | (a) energy ratio, best platform each; (b) latency ratio | (a) at least 1,000; (b) at least 100 | (a) 0.45; (b) 0.75 |
| H15.4 | K1 kernel | identity with `logprob_q` at every validation and H' position; speed against the numpy pass (13,160 ns per token [measured: `results/20260924T102454Z_cvtm_a3_H8_7/`]) | bit-identical; at least 10 times faster; tokens per second per thread reported | 0.95; 0.85; at least 1M per thread 0.55 |
| H15.5 | Cost of TA add-ons | energy per token of CVTM plus each credited add-on | reported in microjoules per token per 0.01 bits per byte | at least doubles the CVTM's: 0.85 |
| H15.6 | Training at equal quality | CVTM-core fit joules (H8.5, idle-subtracted, plus derived host CPU) against E\*(full) of H9.7 (plus its derived host CPU) | at most 1/10 | 0.8 (at most 1/100: 0.1) |
| H15.7 | Today's numpy form | 1-thread numpy CVTM against M2 L6 W256 fp32 on the V100 | lower energy per token | 0.75; lower throughput 0.9 |

**Family H16: distillation pilot** (docs/12 F22; label "distilled"; exploratory, seeds 100 or more; one pilot, not a ladder).
- **Teacher:** the best in-house checkpoint when the pilot starts (the GX tuned model if it exists, else M2 L6 W256), trained only on WikiText-103 train with BPE-8k. Its training joules and all labelling and sampling joules are charged to the student.
- **Arms:**
  - (a) expected-KN soft counts from the teacher's top-32 over the training split;
  - (b) 250M teacher-sampled tokens counted by KN-5 as a component of the section 11.4 mixture, extended to 1B only if (b) gains at least 0.005 at 250M;
  - (c) C-noise: the copy selector trained on teacher-sampled labels against its gold-label twin, with LR and GBDT students on the same targets, judged on H'' residue 5.
- Distilled rows never appear in a teacher-free comparison without the label. H9.9 and H9.11 are not claimed for them.

| Id | Hypothesis | Measurement | Threshold (target) | Prediction |
|---|---|---|---|---|
| H16.1 | Soft counts plus a sampled corpus improve the CVTM | validation bits per byte against the teacher-free CVTM-core | the route continues only if (a) plus (b) gain at least 0.030 with the CI excluding 0; otherwise it is closed and reported | at least 0.030: 0.1; at least 0.010: 0.4 |
| H16.2 | Label noise limits TA learning | arm (c) | credit only if the teacher-label TM beats its gold twin by at least 0.005 AND the best non-TM student by at least 0.002, CI excluding 0 | 0.05 |
| H16.3 | Cost | total joules per 0.01 bits per byte gained | none; reported | as stated |

**Family H17: self-supervised logical representations (docs/12 F23).** Registered separately in amendment A7.

**Overall predictions** [unverified judgement, before any A6 output]:
- a teacher-free Tsetlin-form LM reaches validation bits per byte at most 1.2341 by 2026-10-15: 0.85;
- within 0.02 of the tuned 600 s transformer at equal data: 0.35;
- at most 0.95: 0.02;
- Tsetlin learning is credited with at least 0.010 anywhere in A4 to A6 at full data: 0.2.

**What is NOT claimed under A6:**
- H12 credits counting and copying in clause form. Added data changes the benchmark, so every H12 number carries its added-data size and the equal-data transformer row.
- No "comparable with transformer LLMs" and no "LLM" wording; the C-rule sentence is the only comparability sentence allowed.
- Ratios name the platform, the mode and the accounting. CPU joules are derived. No ratio against a better model is called an equal-quality ratio. Memory is stated beside every ratio.
- PC-TM's concept tree and EM weights are not Tsetlin-learned.
- H13.1 is rule discovery, not LM accuracy.
- Distilled rows include their teacher's joules and never support a teacher-free claim.

### A7 (2026-09-24): logical self-supervised representations with Tsetlin machines (family H17)
**Reason.** The goal of 2026-09-24 (docs/12 F23): teacher-free, scale the Tsetlin machine itself, focus on
"logical self-supervised learning of language" (the proposal title), where what is learned may be patterns beyond
language. A7 registers how the self-supervised (SSL) representations learned by Tsetlin automata (TA) are judged:
by transfer with few labels, by concepts, by a second (image) domain, by the hierarchy of the project proposal, and by
scaling with energy, always against controls that separate TA learning from counting. Nothing above is changed; A3 to
A6 keep every threshold, gate and decision row. Design source: the program design of 2026-09-24 (a scratch report;
its content is summarised here and fixed by this text).

**Disclosure.** Seen before this amendment was saved (20:08 UTC): everything disclosed in A3 to A6; the
exploratory early-test runs `results/20260924T151234Z_ssl_transfer_data_exploratory/`,
`results/20260924T152912Z_ssl_transfer_features_exploratory/` (features only, checks) and the killed classification
runs of 15:47 and 16:37 UTC, whose logs were not read beyond their last lines; from the running
relaunch `results/20260924T200541Z_ssl_transfer_classify_baselines_exploratory/` and
`results/20260924T200544Z_ssl_transfer_classify_banks_exploratory/` exactly four cells were seen:
AG News 256 labels seed 102 LR bag of unigrams and bigrams 0.6692; seed 103 LR bag of unigrams (log1p) 0.6645;
SST-2 256 labels seed 103 LR epoch-1 bank top features 0.5115; SST-2 1024 labels seed 101 LR epoch-4 bank all clauses
0.6594. No concept-test output was seen. SST-2's validation split is being read by those exploratory runs; it is
therefore NOT used for any confirmatory A7 number (below).

**Definitions.**
- **Banks.** L1: the docs/11 engine trained by a self-supervised pretext (masked, causal or diffusion token
  prediction on WikiText-103 train prefixes; masked patch prediction on images). L2: Granmo's AND-OR hierarchical
  package (`PyHierarchicalTsetlinMachineCUDA`), trained greedily on L1 outputs or on bag of words.
- **Codes.** Text code: log(1 + count) of fired clause ids over the positions of a text, L2-normalised; variants "all
  clauses" and "dimension-matched" (the 8,192 clauses that fire most on the pretraining text). Word code: fired-clause
  indicators summed over up to 1,000 training occurrences of the word (all its subtokens hidden), PPMI-weighted,
  cosine. Image code: counts of (clause, image quadrant).
- **Probes.** Primary: L2 logistic regression, C chosen by 4-fold cross-validation inside the labelled sample only.
  Secondary: a tmu Coalesced TM on binarised features (keeps the pipeline TA end to end).
- **Metrics.** Q_text: mean accuracy over AG News, SST-2 and TREC-6 at 32 labels per class, 5 stratified label draws
  (also reported at 8, 128 and 512 per class and full). Q_con: mean Spearman over SimLex-999, WordSim-353 and MEN on one
  common covered pair set (a word is covered with at least 50 occurrences in the pretraining prefix). Q_pred: count-vote
  masked top-1 on the frozen V 8000 validation targets. Q_int: NPMI coherence and intruder score of clause concept
  groups.
- **Learning-credit controls (every arm).** C0y: the same run stopped at 1 percent of epoch 1 (young bank). G+: the most
  frequent single (offset, symbol) literals and literal pairs, equal in number to the clauses. Random kind_map (concepts
  removed from the literals). Count baselines: bag of words, cluster bag of words, PPMI-SVD over the same window of 8,
  CBOW-kNN, KN-5 substitutes. TA learning is credited only by a win over BOTH C0y and G+.
- **Splits.** Exploratory runs select on a dev set carved from each task's train split by a fixed hash (10 percent) and
  never load a test split. Confirmatory: AG News and TREC official test splits, one access per frozen configuration,
  counted like the WikiText-103 test; SST-2: its validation split has been read by exploration, so the confirmatory SST-2
  set is a hash-carved 5 percent of SST-2 train (docs/11 section 3 mix64 of the example index, residue 0 mod 20),
  excluded from every confirmatory training and label sample. Datasets are hashed once into write-once files
  `docs/notes/preregistration-data-hashes-A7-<dataset>.json`; licences recorded (AG News non-commercial research; SST-2,
  TREC card "unknown"; MEN CC BY 2.0; Fashion-MNIST MIT).
- **Frequency-trap levers** (data only unless stated): F1 target subsampling, keep target p with probability
  min(1, sqrt(t / f(y)) + t / f(y)), t = 1e-4 fixed now, draw by mix64(p); F2 at most M = 2,000 targets per class per
  epoch; F3 `neg_budget` 1; F4 `cand_tries` 4. Engine changes (negatives from unigram^0.75, vote steps proportional to
  gate mass) are NOT part of A7; they need their own specification.

**Phases and gates.**
- R0 (exploratory, existing banks: the masked V 2000 banks, the CP bank at epochs 1 to 4, the young bank). Sets priors;
  nothing stops on it.
- R1 pretext tournament: V 8000, C = 65,536, the first 20M training tokens cut at a document boundary, 4 epochs; arms A0
  default masked, A1 masked + F1, A2 F1 + F3 + F4, A3 F2, A4 causal, A5 diffusion, A6 A1 with a random kind_map; B1 slide-11
  pairs (package, hierarchical and flat, bag of words); seed 100 first (exploratory), then seeds 1 to 3 for the top two
  arms and every control. Winner = highest Q_text; ties within 0.5 points go to Q_con.
- R2 hierarchy: H-AO synthetic mechanism test, then H1 (package on bag-of-words pairs, the registered H4.1 task) and H1'
  (the package over the R1 winner's 4,096 most-fired clause bits).
- R3 scaling: the winner (and A0) over C in {65,536; 262,144; 1,048,576; 4,194,304; 16,777,216} at 20M tokens and over
  {5M, 20M, 67M, 135M} tokens at C = 1,048,576; NVML joules per example under the exclusive lock (CPU slot taken first,
  docs/12 F25); a from-scratch masked transformer (L6 W256) with a joule cap equal to each bank's training joules as the
  neural placement. A point projected above 12 GPU-hours runs 1 epoch and is labelled.
- R4 images: Fashion-MNIST train split only (60k; a hash-carved 10k dev; test never loaded), binarised at > 75, 4x4
  patches coded to V = 2,000, 56 tokens per image, masked patch prediction with F1, C = 65,536.
- G-R1 after R1: pass (H17.1) opens R3 with the winner; fail but at or above bag of words: R3 runs as a cost-and-form curve
  with no TA claim; fail and below bag of words: one text curve only, weight moves to R2 and R4.

| Id | Hypothesis | Measurement | Threshold (target) | Prediction |
|---|---|---|---|---|
| H17.1 | TA learning adds representation value (learning credit) | Q_text of the R1 winner minus max(C0y, G+), seeds 1 to 3, paired bootstrap over label draws and seeds | at least 1.0 point, CI excluding 0 | 0.30 |
| H17.2 | SSL clauses beat bag of words | Q_text of the winner minus the bag-of-words baseline (same probe) | at least 2.0 points, CI excluding 0 | 0.40 |
| H17.3 | SSL clauses encode concepts (C1) | Q_con of the winner minus PPMI-SVD on the same covered pairs | at least 0.0 (at or above PPMI-SVD), and above C0y by at least 0.05 | 0.25 |
| H17.4 | The data-level fix of the frequency trap works | Q_text of A1 minus A0 | at least 1.0 point, CI excluding 0 | 0.45 |
| H17.5 | Granmo's AND-OR hierarchy is needed on the mechanism test (H-AO) | held-out accuracy, 5 seeds; hierarchical package vs the same package flat | hierarchical at least 0.90 AND at least flat + 0.20 | 0.60 |
| H17.6 | Two-level logical SSL (H1') beats the registered H1 | slide-11 pair accuracy, 5 seeds, paired Wilcoxon | H1' at least H1 + 2.0 points, one-sided p below 0.05 | 0.35 |
| H17.7 | Representation quality scales with clauses | Q_text (dimension-matched and all clauses) from 65k to 1M clauses | non-decreasing within the seed range at each step | 0.55 |
| H17.8 | Representation quality scales with pretraining data | Q_text at 5M, 20M, 67M, 135M tokens | strictly increasing | 0.60 |
| H17.9 | At equal training joules the TM beats a masked transformer as a representation | Q_text, TM bank vs the joule-capped masked transformer's mean-pooled features, same probe | TM higher; reported both ways | 0.30 |
| H17.10 | The same engine learns patterns in images | image code minus max(C0y image bank, bag of visual words) at 10 labels per class | at least 2.0 points, CI excluding 0 | 0.40 |
| H17.11 | The probe can stay a TM | Q_text with the Coalesced TM probe minus the LR probe, winner | at least -2.0 points | 0.50 |
| H17.12 | Learned concepts are coherent | Q_int (NPMI) of the winner minus C0y | above, CI excluding 0 | 0.45 |

**What is NOT claimed under A7:** no "LLM" or "resolved" wording; no credit to TA learning without a win over C0y AND
G+; no scaling claim beyond the measured range; R0 and every seed-100 run are exploratory; the design and thresholds
were set after the LM results of A3 to A6 but before any R1 to R4 output and before any R0 verdict.

### A8 (2026-09-25): stronger learning-credit controls, hierarchy diagnostics, energy comparators and speedup policy (additions to A7, A4 and A6)
**Reason.** The decisions of 2026-09-25 (docs/12 F28): aim for accuracy with little or no training and judge it on
evidence; cut compute only by quality-neutral techniques; review the project as the supervisor would; check the literature before
expensive runs; keep the codebase clean. Three findings made before this amendment motivate it. (1) The A7 control C0y is cut
at 1 percent of epoch-1 BATCHES, so arms with shorter epochs get a weaker control: A3's C0y saw 78,848 examples with 34,408
live clauses, A0's 200,704 examples with 58,029 [measured: `results/20260924T215716Z_ssl_r1_train_A3_seed100_exploratory/`, `results/20260924T211347Z_ssl_r1_train_A0_seed100_exploratory/`], and the 32-label credit of the
four arms probed first ranked in exactly that order. (2) A verified literature sweep (../../docs/references.md,
docs/09 additions) predicts that most A7 rows fail and gives mechanisms (s and T settings, noisy next-token labels that never
let TA states absorb, long negated clauses, class balance). (3) Under A7 as written, H17.5's conjunct "at least flat + 0.20"
cannot be met once the flat control scores 0.80 or more. Nothing above is changed: every threshold, gate, decision row and
registered measurement of A1 to A7 stands, and no registered row is cut.

**Disclosure.** Seen before this amendment was saved (2026-09-25 07:25 UTC): everything disclosed in A3 to A7; the R1
seed-100 probes of A0, A1, A3, A4 and A5 (complete) and A2 (partial, 383 cells; its C0y at 32 labels was reported
as 53.31, partial); the R1 baselines; the A7 R0 and R4 exploratory results; the H-AO pilots on c85ca4e7 and HEAD
5b4a223; the confirmatory H9 verdicts (`results/20260925T063437Z_cvtm_copy_selector_H9_verdicts/`: H9.1 met, H9.2 trails, no
D0.3 credit, row D7); the GPU LR equivalence run (`results/20260925T055129Z_ssl_r1_lr_equivalence_A3/`, exit 1). The
exploratory control study of this amendment (`experiments/ssl_r1_controls.py`) had started two training runs at 07:22 UTC;
no output of it was read. Its reading rule is fixed in `experiments/cvtm/configs/ssl_r1_controls.json`.

**1. Learning-credit controls for the confirmatory R1 seeds (additions to H17.1, H17.3, H17.10, H17.12).**
- Every confirmatory R1 training run (seeds 1 to 3) also saves the young ladder of the control study: epoch-1 checkpoints after
  batches {77, 196, 255, round(5 percent), round(10 percent)} plus the registered C0y. Batch 196 of 1,024 examples equals
  A0's C0y exposure (200,704 examples) and is the exposure-matched young bank.
- For the top two arms, a literal-frozen twin is trained on each confirmatory seed: the same arm with `p_ins = 0`,
  `p_neg = 0`, `s = 2^62` (every literal-changing path of docs/11 section 5 disabled; seeding, recycling and vote learning on).
- H17.1 keeps its registered definition and verdict (winner minus max(C0y, G+)). Reported next to it, with the same probe
  and the same statistics: winner minus the exposure-matched young bank; winner minus max(young ladder of the same run);
  winner minus the best young bank of any arm; winner minus the twin. A new secondary label, "literal-learning credit", is
  given only if the winner beats max(young ladder) AND the twin by at least 1.0 Q_text point at 32 labels, all-clauses AND
  dimension-matched codes, with the CI excluding 0. No other wording credits Tsetlin literal learning.
- Statistics for H17.1 and the rows above: a two-level bootstrap (resample seeds, then label draws within each resampled
  seed; 10,000 resamples), reported with the seed-level Student-t interval. This specifies how "paired bootstrap over label
  draws and seeds" treats draws nested within seeds; it changes no threshold.
- Added few-label controls, reported and never part of a threshold: multinomial naive Bayes and NBSVM probes on bag of words
  (Wang and Manning 2012), and a random conjunction bank (the same number of clauses, 1 to 3 position-bound literals drawn with
  the pretraining text's literal frequencies, no training) as a third learning-credit control.
- Q_pred (defined in A7, not yet computed in R1) is computed for every confirmatory bank, its young ladder and its twin with
  the T1 count read-out on the frozen V 8000 validation targets, as `experiments/ssl_r1_controls.py --stage qpred` does.

**2. Exploratory configuration arms before any A7 conclusion about Tsetlin learning** (seed 100 first, then 101; never
confirmatory). The R1 settings (s = 128, T = 64, budget 12, neg_budget 1) were tuned for the tiny-LM gate KB2, not for
representations; the literature puts text optima at s of about 1 to 11 and 1 to 4 literals per clause (Tarasyuk et al. 2026;
Abeyrathna et al. 2023, CSC-TM). Arms on A3's data: s in {8, 32}; `budget = 3` with `p_neg = 0` and `neg_budget = 0`
(positive short clauses, the slide-11 setting); each with its own young ladder and twin, the same probes and Q_pred, and the
per-run diagnostics of item 6. They inform the design only; no A7 threshold is read on them.

**3. H17.5 and R2 diagnostics.** Stated before any confirmatory H17.5 seed: with the flat control at 0.80 or more (exploratory
0.815 to 0.829), the conjunct "hierarchical at least flat + 0.20" cannot be met; H17.5 is run and reported exactly as registered
(package c85ca4e7, docs/12 F26). Added exploratory rows on the H-AO world (world seed 100, then 101): (i) the positive-literal
pair (append_negated False), where a flat machine cannot express the concept compactly; (ii) accuracy on seen and on unseen
combinations separately; (iii) oracle-initialised stability: the true hierarchy written into the package state, verified by
read-back and prediction, then trained without re-clamping at s in {4, 10, 25}, accuracy and per-node recovery every 5
epochs; (iv) an epochs-to-threshold curve on one instance with the package demo's scale in mind (its noisy-parity example
reaches 99.71 percent only after about 7,780 epochs [verified in source: third_party/ref/PyHierarchicalTsetlinMachineCUDA
README, 778 printed rows of 10 epochs]), capped at 2 GPU-hours; (v) Valiant elimination (Valiant 1984, Theorem A): one pass over
positive bags, a frequency prefilter, ORs of up to 6 words deleted when a positive misses them, as a logic-native non-TA
learner of the same AND-of-OR concept; (vi) Granmo's own ANDOfXORsDemo and MultiConceptDemo at his defaults as positive
controls first. A width, distractor and sample-size factorial (vocab {24, 200, 2000} x distractors {0, 24} at 8,000 fit bags)
is added if (iii) shows the true hierarchy is stable at some s.

**4. The slide-11 pretext (H4.1 to H4.3, H17.6).** Added exploratory variants next to the registered rows: negatives that
swap paragraph order within a document and topic-matched negatives from other documents (a same-or-different-document
objective learns topic, ALBERT's NSP 90.5 percent on its own task vs 52.0 on sentence order [verified in paper: Lan et al.
2020, Table 5]); a bag-of-words overlap baseline on the pairs; OR-group synonym recovery against WordNet synsets.

**5. Energy and memory (H8.5, H9.7, H12.5, H15).**
- H8.5 has no valid measurement yet (`results/20260924T104118Z_cvtm_H8_5_fit_energy/` stopped after 3 repeats;
  `results/20260924T151319Z_cvtm_H8_5_fit_energy/` took its idle baseline right after another GPU job, median -207.9 J). It is
  rerun under the exclusive lock with the idle baseline measured before AND after each repeat and reported both ways, plus a
  CPU-only fit row (joules derived from CPU seconds by the A6 E-rule).
- Every energy row reports raw GPU joules, idle-subtracted GPU joules, derived CPU joules and resident bytes. Timed passes last
  at least 5 s with at least 4 trials (NVML on the V100 samples power in 10 ms windows every 20 ms).
- A plain n-gram comparator is added to H15: the same KN-5 exported as ARPA and queried by KenLM built query-only if it builds
  here without cmake, otherwise by an in-house plain KN C lookup; the TM form's cost is stated relative to it ("logic tax").
- Memory: a bits-per-byte against resident-bytes curve with lossless count structures (exact log-probabilities checked
  bit-identical on validation) and the H8.9 pruned variants labelled lossy.

**6. Per-run diagnostics** (the CAIR group's quantities, reported for every trained bank): histogram of included literals per
clause, negated share, bag-literal share, owned clauses per class against T, the fraction of TA transitions that cross the
include boundary per epoch, and class-sum histograms.

**7. Speedup policy.** A speedup is used for a registered row only if it reproduces the registered estimator on logged cells
under a rule written before the check. The GPU LR backend failed such a check (identical accuracy in 331 of 360 cells) and is
not used. Lossy speedups (absorbing exclude states, drop clause, literal sampling, class-sum-free updates, early elimination of
configurations, count pruning) enter only as separately labelled rows with their measured quality cost. Registered rows that
the literature predicts to fail run at their registered minimum, cheapest first, with no extension seeds or arms.

**8. Wording.** The Track A "exact Binomial draw" novelty is narrowed to the negative-class loop: the Weighted TM already drew
the number of Type I recipients from a Binomial and then chose them uniformly (Phoulady et al. 2019, section 3.1). CAIR prior art
listed in docs/09 (2026-09-25 additions) is cited, not claimed: self-supervised TMs (Label-Critic TM 2022), TM word embeddings
(TM-AE, Omni, FastOmni), the SAT encoding of TMs (Przybysz et al. 2023), TM uncertainty scores (Helin et al. 2025), position
literals (Convolutional TM 2019). H17.9's transformer is named "plain MLM" with its recipe stated.

**What is NOT claimed under A8:** a reading of the exploratory control study or of the configuration arms is not an A7 claim;
"literal-learning credit" needs item 1's rule on the confirmatory seeds.

**A8 erratum (2026-09-25 08:20 UTC, before any output of items 2 and 3).** Item 2 lists the R1 settings as "neg_budget 1":
arm A3 and every R1 arm except A2 use neg_budget 2 (A2, lever F3, uses 1). Item 2's "each with its own ... twin": one
literal-frozen twin, A3T of the control study, serves all three configuration arms, because each arm's twin would reproduce
A3T (the s arms give an identical engine config once the twin overrides apply; for the budget-3 arm the trajectory is identical,
since budget and neg_budget are read only when a literal state crosses 127 to 128, which a twin never does; checked bit for bit
by `tests/test_ssl_r1_config_arms.py`).

### A9 (2026-09-25 12:25 UTC): readings for the confirmatory R1 evaluation, fixed before any confirmatory output
**Reason.** The confirmatory R1 driver (`experiments/ssl_r1_confirm.py`, config `experiments/cvtm/configs/ssl_r1_confirm.json`)
exposed readings that A7 and A8 leave open and that decide verdict inputs; a config file is not a registration (protocol rule 16), so
they are fixed here. **Disclosure:** everything disclosed in A8, plus the complete R1 seed-100 report
(`results/20260925T105202Z_ssl_r1_report_seed100_exploratory/`); no confirmatory output exists and no confirmatory test split
has been accessed (the test-access log holds 22 lines, all older). Where docs/12 already decided a reading before any R1
output, that decision stands even where the seed-100 data would now favour another reading.
1. **Winner:** A3, fixed by the seed-100 tournament (A7 "Winner = highest Q_text"); A2 is the second arm. Seeds 1 to 3 confirm;
   they do not re-select.
2. **Code that decides H17.1, H17.2, H17.3 and H17.4:** the all-clauses code, as docs/12 F26 (2) decided on 2026-09-24 before
   any R1 output. That decision's stated reason ("exactly equal feature counts") is nominal only, because C0y has fewer live
   clauses; the A8 exposure-matched, ladder and twin rows exist to correct for that, and the dimension-matched code is reported
   for every row. On seed 100 the choice matters (A3 minus max(C0y, G+) is +2.83 all-clauses and -1.18 dimension-matched), so
   both are always printed side by side.
3. **H17.11:** TM probe on the dimension-matched binary code minus the LR probe on the SAME dimension-matched code (the probe is
   the only difference); TM (dim) minus LR (all clauses) is reported.
4. **max(C0y, G+):** per seed, the control with the higher mean confirmatory Q_text, fixed before resampling.
5. **Statistics:** the A8 two-level bootstrap (seeds, then draws within seeds; 10,000 resamples) decides "CI excluding 0" for
   H17.1, H17.2 and H17.4; the seed-level Student-t interval is reported. The A8 rows "minus the exposure-matched young bank" and
   "minus the best young bank of any arm" are estimates with intervals and NO registered threshold (no MET or NOT MET).
6. **Q_con (H17.3):** SimLex-999 and WordSim-353 in full and MEN in full (its 2,000 development and 1,000 test pairs), on one
   common covered pair set; the MEN test pairs are loaded through the counted path, once per configuration. H17.3 is read on the
   mean over seeds 1 to 3.
7. **Q_int (H17.12):** a clause's concept group is the top 10 words by PPMI with the clause's firings (at least 5 counts),
   over the 8,192 dimension-matched clauses; NPMI over a 10-token window with the WikiText-103 train tokens AFTER the 20M-token
   pretraining prefix as the primary reference (unseen by every bank; the validation split, where only about 2 percent of pairs
   co-occur, is the secondary reference).
8. **Full budget (A7 "and full"):** Q_text at the full label pool for A3 and A2 (epoch 4, C0y, G+, bag of words), draw 0 as in
   R1, on every confirmatory seed.
9. **Random kind_map control for the winner:** A3 with a random kind_map (A3R, one fixed token permutation of the three cluster
   levels, as A6) is trained on seeds 1 to 3 and probed with the registered probe set; reported, not part of any threshold.
10. **Test files:** before the first counted access, the sha256 of each confirmatory test file is taken from the dataset host's
    published file metadata at the pinned revision (no content download) and written to a write-once file; every counted access
    verifies the downloaded bytes against it. One counted access per (configuration, task): 28 configurations (the driver's 25, that is seven arms on three seeds, the
    common baselines and three random banks, plus A3R on three seeds) x 3 tasks, plus one MEN-test access per Q_con
    configuration; any second access needs a logged reason and is reported.
11. **Launch condition:** the confirmatory chains start only after the exploratory control-study and configuration-arm chains
    have finished (their Q_pred and census stages assert that the shared test-access log does not change during a run).

### A10 (2026-09-25 14:11 UTC): reference digest for confirmatory test files stored in git rather than LFS
**Reason.** A9 item 10 requires the sha256 of each confirmatory test file from the host's published file metadata before any
access. The Hugging Face host publishes an LFS sha256 for the AG News test file (1,234,829 bytes) but not for the TREC test file
`default/test/0000.parquet` (17,084 bytes), which is stored in git: its metadata carries only the git blob id (a SHA-1 over the
header "blob <size>" and the file content), as `docs/notes/preregistration-data-hashes-A6-trec.json` already recorded for the
TREC train file (`hub_lfs_sha256: null`). No confirmatory output exists and the test-access log still holds 22 lines, all older.
**Rule.** For a file whose published metadata has no LFS sha256, the pre-access reference is the published git blob id plus
the file size, written to the same write-once file with the field names saying so; every counted access recomputes the git
blob SHA-1 of the downloaded bytes and checks it and the size before use, and logs the sha256 of the bytes as well. Files with a
published LFS sha256 keep the A9 rule. Nothing else changes.
