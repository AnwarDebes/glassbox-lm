# REGISTRATION T5: word codes learned by a Tsetlin machine autoencoder, in place of potion (val; written 2026-10-18T09:06:44Z)

Why: the potion table is distilled from a neural network (bge-base), so "no neural network at any stage" cannot be
claimed for potion bits (the study log, CLAIMS AND CITATIONS). Here a Tsetlin machine autoencoder (TM-AE; Bhattarai et al.,
arXiv 2301.00709) learns the word codes, and the sentence bits stay a lookup plus OR (R1) or go through the same
existsT encoder (R3a). No published numbers exist for TM-AE bits as direct TM input (CLAIMS item 5). Expected (not a
measurement): above words only, below potion; the count-only twin decides whether the TM adds anything over counting.

This is a VAL comparison. No test split is read. If H5a or H5b works, a new registration fixes a test run.

## Fixed

- Corpus (a): t5/corpus_a.npz (corpus npz sha256 3e301f509c5a7d0c..., from build_corpus_a.py sha256 84b5de842a961d67):
  the train texts of CLINC150, Banking77 (study split) and HWU64 plus the CLINC oos_train, minus every text whose
  normalized hash equals a val or test text of any evaluation split (1,841 dropped), as sets of potion-base-32M tokens;
  vocabulary = the 4,564 tokens with document frequency >= 2; 31,196 texts. No MASSIVE text.
- TM-AE (one run, CPU, one thread): tmu 0.8.3 TMAutoEncoder through t5/tmae_fixed.py MaskedClassSumAE (the paper's
  masked class sum, Eq. 4), 600 clauses, T 1200, s 5.0, accumulation 25, no negated literals, output_balancing 0.5,
  every vocabulary token both a target and a feature, 2,000 rounds (the paper's r), seed 1 with the libc srand fix.
  Checkpoints (weights W and include matrix) every 250 rounds. If 2,000 rounds are not done after 10 h, the latest
  checkpoint is used, and that is reported.
- Readouts, per vocabulary token w (tokens outside the vocabulary get no bits and no vector):
  - R1 (primary, pure logic): bit j = [W[w, j] >= 2] (2 and not 1, because tmu starts every target at the same random
    +-1 weights; pilot), 600 bits; the sentence's bits are the OR over its tokens.
  - R3a (primary): the row clip(W[w], 0, T) * [W[w] >= 2], L2-normalized, through the existsT encoder of
    exists2_tm.py: 512 concepts by spherical k-means (seed 0, 25 iterations) on the codes of the dataset's training
    tokens, per concept the max cosine over the sentence's tokens, an 8-level thermometer over the training sentences.
- Controls, same tokens, same harness:
  - potion-V: the potion-base-32M vectors (revision 1e5a03f8) of the same 4,564 tokens (other tokens: none), through the
    same encoder at K = 512 (bit-matched to R3a) and K = 86 (about 600 bits, bit-matched to R1).
  - PPMI-V (count-only twin): PPMI rows of the token-by-token document co-occurrence counts of corpus (a), context
    smoothing alpha 0.75, 4,564 x 4,564, through the same encoder at K = 512 and K = 86.
  - words-V: one bit per vocabulary token present (4,564 bits).
- Classifier: the decision-layer CoTM with the settings of exists2_tm.py (4,000 clauses, T 4000, s 8, 40 epochs, batch
  128, hardest negative, negation, init half), val accuracy at the best val epoch (as every exploratory row of tests
  1-3), seeds 1 and 2.
- Datasets: CLINC150, Banking77, HWU64 (val splits). MASSIVE is excluded: corpus (a) holds HWU64 text, and HWU64 and
  MASSIVE share many utterances (the study log, DATA FINDING).

## Bars (mean of seeds 1 and 2, val accuracy in points, per dataset; G = potion-V K512 - words-V)

- H5a works (R1, pure lookup + OR): R1 - words-V >= 0.25 G on at least 2 of 3 datasets, AND R1 - PPMI-V K86 >= +0.5
  on at least 2 of 3. Strong: R1 - words-V >= 0.50 G on at least 2 of 3 (and the count bar).
- H5b works (R3a): R3a - words-V >= 0.25 G on at least 2 of 3, AND R3a - PPMI-V K512 >= +0.5 on at least 2 of 3.
  Strong: >= 0.50 G on at least 2 of 3 (and the count bar).
- Reported, no bar: potion-V K86; every view against the full potion existsT val rows of test 2 (the study log 15:54:59Z,
  which use each dataset's own vocabulary); share of val tokens without a code; bits on per sentence; the TM-AE
  training time and weight histogram; probe neighbours; readable clauses of R1 bits (a bit = a clause of context words).
- Not run (pilot: degenerate codes): the Omni recipe and the literal-count rows (R3b). Nothing else is tried on these
  val splits before this verdict; anything else is labelled exploratory.

## Run

At most 2 jobs of this line. The TM-AE training starts in the slot that T4 lane B frees (about 12:40Z); the control views
(no TM-AE needed) run in the slot that T4 lane A frees; R1 and R3a run after the TM-AE training. Code hashes of the
training, code-building and harness scripts are added below before their first run.

## Amendment 1 (2026-10-18T09:20:38Z, before any val run; only smoke runs on a slice of the TRAIN split have run)

- The count-only twin becomes PPMI-SVD: the PPMI matrix of the registration (document co-occurrence on corpus (a),
  context smoothing alpha 0.75, diagonal set to zero: a token does not co-occur with itself), reduced by truncated SVD
  to 600 dimensions, rows U_600 * S_600^0.5 (eigenvalue weighting 0.5), L2-normalized, through the same encoder at
  K = 512 and K = 86. Reason (smoke on banking77 train sentences): the raw PPMI rows are sparse and non-negative, so
  many thermometer thresholds fall on 0 and merge, giving 311 bits at K = 86 (R1: 600) and 2,152 at K = 512 (R3a:
  3,509); the twin would not be bit-matched and would be a weak baseline. PPMI-SVD is the standard dense count model
  (Levy, Goldberg and Dagan 2015) and has the TM-AE's 600 dimensions. Raw PPMI rows are not run.
- Bit counts are reported for every view; the bars compare views at the same K (same number of concepts), as
  registered. If a twin's bit count on the training sentences is more than 15% away from its TM-AE view, that is
  reported next to the verdict.
- Order of runs: wordsV, potionV K512, ppmiV K512, ppmiV K86 first (they enter the bars), potionV K86 last (report only).
- Other choices, confirmed: a sentence with no coded token gets max-cosine 0 (as exists_tm.pooled); the TM-AE runs as
  200 fit calls of 10 rounds, libc srand re-seeded before each call.

## Code and data hashes (2026-10-18T09:27:02Z, before the first registered run; sha256, first 16 hex)

- t5/corpus_a.npz 3e301f509c5a7d0c (build_corpus_a.py 84b5de842a961d67); tmae_fixed.py bb179b9f2562524d;
  train_tmae.py 9c0ca84d0cef7373; codes.py a414e651ced287b4 (the control tables were built at 09:23:53Z (file time) by its previous
  version f91a849b5621287f; the only change since is an atomic write of the manifest, so a run never reads a
  half-written manifest); t5_tm.py 45f9e733575e6273; t5_queue.sh 55722228e0dcbcf5 (it checks t5_tm.py, codes.py and
  dl/tm.py, tm_init.py, boolean.py, data.py before every run).
- Control tables: codes_wordsV.npz 9a6d694fdf4a933a, codes_potionV.npz 000c40c693e25a25, codes_ppmiV.npz
  21717cba2d87aa7b (PPMI-SVD, 4,564 x 600; 37.2% of the singular-value sum kept, 64.5% of the squared sum).
- Smoke bit counts (banking77, thermometer on 2,000 training sentences, R1 / R3a from a 20-round stand-in): R1 600,
  R3a 3,509, potionV 3,329 (K512) / 565 (K86), PPMI-SVD 2,795 (K512) / 452 (K86). The PPMI-SVD twin has 20-25% fewer
  bits at the same K, beyond the 15% line, so this is reported with the verdict. The missing bits are thresholds that
  coincide (identical thresholds give identical bits), so the merging itself removes no information.
- Slots: t5/slot2_next.sh waits for T4 lane B (slot2_chain.sh), then runs train_tmae.py and t5_queue.sh tmae;
  t5/slot1_next.sh waits for T4 lane A, then runs t5_queue.sh controls. Logs t5/runs_t5.log, t5/runs_tmae.log.

## Verdict script (2026-10-18T13:26:17Z, before any registered val row: t5/results_t5.jsonl did not exist yet)

- t5/verdict_t5.py 0bca1ea37a166299 (self-tested on synthetic rows in a scratch folder, never on real rows). It applies the bars as
  registered and fixes five cases the text left open: the two counts of a bar are separate (they need not hold on
  the same datasets; the datasets where both hold are printed too); a dataset with G <= 0 counts as not met for the
  gap parts and is flagged; the comparisons allow 1e-9 for float rounding; a view counts on a dataset only with both
  seeds (else the bar is printed as incomplete), and a repeated row is ignored (the first is kept).

## Verdict (2026-10-18T18:39:03Z; t5/verdict_t5.py 0bca1ea37a166299, output in t5/verdict_t5.txt; 42 rows, no script or table hash differs)

- H5a (R1, clause bits OR-ed over the tokens) NOT MET: val 45.15 / 31.10 / 48.00 (CLINC150 / Banking77 / HWU64) against
  words-V 82.62 / 79.91 / 79.98: the OR saturates (477-578 of 600 bits on in the average val sentence).
- H5b (R3a, TM-AE weight rows through the existsT encoder) NOT MET: 85.62 / 85.19 / 81.41; above words-V by +3.00 / +5.28
  / +1.44 (28.6% / 47.5% / 14.7% of G, so the gap part holds on 2 of 3), but against the count-only twin PPMI-SVD K512
  (86.06 / 84.50 / 84.58) only -0.44 / +0.69 / -3.16 (the count part holds on 1 of 3). Potion-V K512: 93.12 / 91.03 / 89.78.
- Bit counts (Amendment 1): PPMI-SVD has 21.0% fewer bits than R3a on Banking77 (the one dataset where R3a beats it) and
  31.7% fewer than R1 at K 86 there; elsewhere within 15%.
- Plainly: the TM autoencoder's word codes (paper recipe, 2.19 h on one CPU thread) are no better than counting as TM
  input here, and far below potion. Probe neighbours are partly sensible (flight: airlines, denver, york; transfer:
  deposited, withdrawal, deposit; cancel: stop) and partly noise (french: developed, maintenance); the strongest
  clauses are conjunctions of function words. No test run follows.
