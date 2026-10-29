# R1 [A-Struct v2]: registration of experiment 1 of Part 3 (the Part 3 plan, section 3)

Written 2026-10-21 (UTC). This published copy is edited in wording only. Binding together with the line "R1 [A-Struct]"
in the study log `the study log` (not distributed), which
holds the SHA-256 of the frozen record `A/R1_A-Struct.frozen.json` (written by `A/freeze.py`; it pins the full
SHA-256 of this text and of every file the code depends on), and, since amendment 1 (section 17, before any real
run), the line "R1 [A-Struct] AMENDMENT 1", which holds the SHA-256 of the record frozen again with `freeze.py
--amend`. Paths are relative to `..` unless
they start with `A/` (= `studies/language_part3/A/`).

**Status at the freeze.** No A-Struct stage has run on any MultiWOZ split: no tune-cal, tune-hold, dev or test
number of this pipeline exists. Only the self-tests ran (synthetic corpora in the MultiWOZ file format; the dry run of
every entry point; section 12). Read-only format checks of the MultiWOZ 2.2 TRAIN split were made earlier, at
03:06:00Z-03:06:55Z (study log 03:55:36Z; no file, no model, no score). No GPU was used.

## 0. Sources (SHA-256, full)

- The Part 3 plan (not distributed), e21fb927b9f28e860df9ee309a16e17af75e6371eb0c6a19b49e4d23401a6f3f:
  section 3 and the review amendments of section 11 are binding. This text adds only implementation choices
  (listed in section 15) and fixes the sub-choices the plan leaves open (section 8).
- Proposal `glassbox-lm/studies/language_part3/proposal_A-structured.md`
  24f75db22633f71f7a53ad34c3d55836fd8ccd18a4848bc8caf4e5310418c4fd (binding where the plan does not change it).
- Data, read only, nothing downloaded:
  - MultiWOZ 2.2 `glassbox-lm/studies/decision_layer/data/untrusted/multiwoz22/` (MIT; commit
    fe0c8e65cfcd8462bd33c86e35f21addc84ca82b): `MANIFEST.sha256`
    87d8b2cd220b8f2cd2f9527f2a223b658151a5cb3f3bd810bbcd4c50209e58cc; S0 checks `train/dialogues_001-017.json`
    and `dialog_acts.json` (328f392165e7826db9f827731b14b5cc04e79e9e3c6332bfb192a1ea17f8e9b6) against it.
  - C4 counting table `glassbox-lm/studies/decision_layer/data/tables/c4words_S1000_v1/` (PPMI-SVD, no neural
    network at any stage): `words.txt` 49dae3df4202d9769e377e8ecfcd4adeffc799907745cfbcbdca83cb7ccd0a29,
    `vectors.npy` 9263bf09383de22806e320a1988de57816dbd62bfa1cc692b0fb7077e44d267c, `tokenize_ref.py`
    2c3c1aff44ed34e017276221dff31ea29e7f9ab1f8f3a77cd0f193ee36e91803, `MANIFEST.sha256`
    f49b934ad1fe8d520af9f87ef37f145aff5483f1c55068e2c5309747f8bd734e. The a3B table is not used.
- Imported code, read only, never edited: tally `src/tally/__init__.py` e3b0c442..., `floor/__init__.py`
  456f452d1ba1780bf3a1b2faf22fdcdce2a2def03163faa81ef693dc564266ab, `floor/kn.py`
  fcdf25e5b7f080737aac5295cb0c87993ce4bcaaaa72a14d411143ea2d3ccd80, `floor/kenlm.py`
  871c8d309f6eb024e58365ede67b74cf321983c1b3d42bcc08013679b4177ae3, `floor/mix.py`
  29f05bb4c4edaec96aa15ecf590b78729aa8a336b0c5fa9dad8e748ddc631b8a, `floor/stats.py`
  64a4efd9f2a83ab94dd1e07611ac340fac66af29e4f3c67d6ffbe1eb1e005b05, `floor/aa.py`
  044022c4074d6355ffeb91ad4573cd0fff054b11bbf235dc62b41f0eabaf1f8c; decision_layer `src/dl/tm.py`
  f649043f5801605b35174c526b290a0cfbadbdb66ade5bf04dafc122289d3424, `src/dl/tm_init.py`
  557ff697065297387c0b698ec77ae1159496e10fd9c1547dc79dbf2ee67ef577, `src/dl/cpu_tm_fast.py`
  bf1c4a9086d76448d9d77cf7775e0bb0e5f4bffc926d777b867f89148279423e; KenLM `tools/kenlm/bin/lmplz`
  cdc75fa471da385d4dffa58836fe4068a60bfe1ebc8bdda76aed645f92f61d75, `build_binary`
  86d0c8432007d3e63490872a3bba89efdb957b15ec96942f49c0530a02ca553f, `query`
  97374c0f9586b0e830090a4f0e21edbfbd03bb6d9596aeeff5a174fe8ec74c45. tmlm's `ngram.py` is not used.
- A's own code: every file under `A/astruct/`, the scripts and shell files in `A/`, `A/tests/`; their full hashes
  are in the frozen record and `A/MANIFEST_R1.sha256`. Every real run (outputs under `A/out/`) stops before writing
  anything unless every enforced file still has its frozen hash and the study log holds the frozen record's hash.
- Recorded in the frozen record, not enforced (an edit for another arm must not block A): the plan, the proposal,
  the evidence and literature files, and the CPU gate script `glassbox-lm/studies/decision_layer/scripts/cpu_load.py`
  2e300054a9284ef4bb0d395048fb5988d3176c100aa184384d22a3cc5eb04328; since amendment 1 also the chain wrapper
  `glassbox-lm/studies/language_part3/chain/part3_cpu_job.sh` and its helper `chain/part3_status.py` (their hashes are in the
  record and the amendment line).
- Environments: `.venv-tm` (python 3.11.16, numpy 2.4.6, scipy 1.17.1, scikit-learn 1.9.1, numba 0.67.0) for every
  counting and sklearn stage; `.venv-dl` (the same numpy, scipy and scikit-learn; torch 2.6.0+cu124, used on the CPU)
  for the K3c pilots. The versions are also written into the frozen record.

## 1. Question and the comparisons that carry the claim

Given a MultiWOZ 2.2 system turn's dialogue acts and the words already written, does a TM that answers "is this
candidate the next word?" beat the best equally tuned conditional counting system B and a counted decision tree on the
same bits, in held-out bits per token? Two co-primary arms: **A-full** (all 364 bits) and **A-nf** (the 320 bits that
are not count-derived). Each arm's headline, fixed now: its TM mix (10-seed mean of per-position bits) minus the better
of B and the arm's tree-twin mix, on the final split (the official MultiWOZ dev split), with the PASS conditions of
section 9. Either arm can pass; both are reported side by side, whatever the result. There is no combined line, and
no claim rests on the other arm.

## 2. Data, splits and split uses

- Replies: every SYSTEM turn of `train/` whose `dialog_act` is not empty (56,581 in train by the format check).
- Split by h = SHA-256 of the UTF-8 `dialogue_id`, read as a big-endian integer, mod 100: **fit** h >= 12;
  **tune-cal** h 0-5 (every choice, every calibration, every EM weight); **tune-hold** h 6-11 (every gate; every
  pick reported unchanged). **Training rows** of the learned arms: fit dialogues with h 12-33 (22 of the 88 fit
  residues, 25%).
- **Final** = the official dev split (1,000 dialogues), opened once by the frozen final script after line R2 (K6). No
  word-prediction model was fitted or scored on it before. Earlier uses, all by the decision-layer study: e77-e79
  (dialogue turns as inputs, the turn's domain-act set as the label, for act prediction) and the e77 contamination
  check (C4 text against the dev utterances: 2.47% matched, all stock phrases). No model of the system-reply words was
  fitted or scored on it.
- **Official test: never read** (it stays for the decision-layer study's R24, which waits at its gate). No A code
  names the dev or test folder except the before/after folder listing (`ls` of the folder, never its files); a
  static self-test asserts this.
- `dialog_acts.json` holds the acts of every split in one JSON object. S0 parses it and keeps only the train dialogue
  ids; no field of another entry is used (the decision-layer study's `multiwoz_prepare.py` reads it the same way).
- One registered final read for this experiment; no other Part 3 registration uses the MultiWOZ dev split (C uses
  PTB; B's proposal uses splits of the train files).

## 3. S0: replies, tokens, vocabulary, inputs (`astruct/data.py`, `s0_data.py`)

- Delexicalisation: the turn's `span_info` spans, sorted by (start, -end, act, slot); a span that starts before the
  end of the last kept span is skipped (the 11 overlapping spans of train, all identical ranges); a span with an
  empty or out-of-range character range is skipped; a kept span becomes `[domain-slot]` (domain = the act name before
  "-", lower case). The character range is used, also where the span's value differs from the text (392 spans). A
  value with no span (3.3% of the act values in proposal A's count) stays as words.
- Tokens: the text between kept spans goes through `tokens()` of `tokenize_ref.py` (never `ids()`); each placeholder
  is put back whole as one token; then the end token `<end>`. Token-count self-check on every reply: the count equals
  `tokens()` of the raw delexicalised reply with each placeholder counted once.
- Vocabulary: `<end>`, `<unk>`, every placeholder type seen in fit (sorted), then the fit word types with count >= 2
  (by count, then string). Every other token, input or target, is `<unk>`. BOS and PAD are context-only symbols.
- Inputs, from `dialog_act` only (self-check on every turn: identical with `span_info` deleted): the domain-act set,
  the (act, slot) pairs (slot lower case; "none" slots included), and one expected placeholder per (act, slot, value)
  whose value (stripped, lower case) is not "none", "?" or empty, kept when its type is in the vocabulary (the plan's
  literal rule: 94,023 in train against 93,710 if de-duplicated by (placeholder, value); 273 turns differ).
- Condition symbols: S = the sorted act-slot set, A = the sorted domain-act set (64-bit hashes of the canonical
  strings, collisions asserted absent). The domain-act columns are every act name of the train split's replies
  (inputs, never targets; 35 by the format check); all other lists below come from fit only.
- Word lists (fit only): top32 = the 32 most frequent words (by token count); cue40 = the 40 words in the most fit
  replies; pairs64 = the 64 (act, slot) pairs in the most fit replies. Ties: string order.
- Concepts (C4 table, fit vocabulary only): the L2-normalised C4 rows of the vocabulary words that have a row;
  spherical k-means (k-means++ start, at most 25 Lloyd steps) with k 32 (seed 0) for the previous token and k 64
  (seed 1) for the candidate; a token's concept = its nearest centroid; a token with no row (and every placeholder,
  `<end>`, `<unk>`, BOS, PAD) has all concept bits 0.
- The frozen bit list `A/out/s0/bit_list.json` (names, the count-derived flag of every bit) is written by S0; its
  SHA-256 goes into the study log (`A/log_bitlist.sh`) before the first tune-cal score; S1 and S2a refuse to start without
  it.

## 4. Positions, candidates, bits, cross-fitting (`positions.py`, `floors.py`, `rows.py`, `rates.py`, `build.py`)

- Every reply position t (words, placeholders and the end token) sees only tokens < t: the history (BOS at the
  reply start, PAD before it), the due placeholders (the expected multiset minus those emitted before t), "said
  before" lookups. No-future check (S2a, 200 random tune-hold positions, seed 0): the reply cut after t, the reply with
  every later token random, and the reply with the token at t replaced give identical candidates, floor numbers and
  bits at t (the last one except the label and the true token's probability); then the same for every learner's
  scores and bits (S2b, K3c).
- Candidates C: the floor's top 8 (ties: smaller id), then the due placeholders not among them (by floor probability,
  ties: smaller id), then `<end>` if not yet in. Label: the candidate is the true next token.
- Bits (364; the 44 count-derived ones marked, absent from A-nf). Per position (226): domain-act present (35);
  the pairs64 present (64); due placeholders >= 1, 2, 3, 4; previous token = top32 word k (32); previous token's
  concept (32); previous token is a placeholder; cue40 word k said before (40); position >= 1, 2, 3, 5, 8, 12, 17,
  25; [count-derived] floor top-1 probability >= .1, .2, .4, .6, .8; [count-derived] floor entropy (bits) >= .5, 1,
  2, 3, 4. Per candidate (138): [count-derived] rank 1-8 or extra (9); [count-derived] log2 floor probability >= -1,
  -2, -3, -4, -6, -8, -11; [count-derived] floor probability / top-1 >= .5, .25, .1, .05, .02, .01; type flags (due
  placeholder, other placeholder, already emitted placeholder, end, punctuation, already said word, equal to the
  previous token, in top32); top32 identity (32); concept (64); [count-derived] rate under the domain-acts >= .01,
  .03, .1, .3, .6, .9; [count-derived] rate under the act-slot set >= the same.
- Counted rates: the share of the counting replies with the condition whose reply contains the word. Domain-acts: the
  maximum over the turn's domain-acts that were counted (none counted: the global rate). Act-slot set S; S never
  counted: the rate under the domain-act set A; A never counted: the global rate.
- **Cross-fitting (review 2).** Folds: whole fit dialogues, 5 folds, tally `aa.folds(seed 1)`. For the fit positions
  of fold f, every count-derived number (candidates, floor probabilities, top-1, entropy, both rates) comes from counts
  on the fit replies of the other four folds (the floor's EM weights refitted on tune-cal for that fold's floor). Tune
  and final rows use the floor and rates counted on all of fit. Leak self-check on the real rows (S2a): every training
  row whose act-slot set occurs once in fit has the act-set rate of its back-off; without cross-fitting the true word
  of such a row would get rate 1 (counted and printed as the negative control). Amendment 1 (review issue 9):
  for the same positions, every S-keyed context lookup of the fold's floor (every level of an F1 chain, the first
  level of F5) is unseen and S's count in the fold's counting replies (the F1 / F6 bucket) is 0; the self-tests run
  the negative control (the floor counted on all of fit finds them).
- Twin rows: every fit position (cross-fitted the same way, about 8.5M rows). Training rows of the learned arms: the
  positions of the training dialogues (about 2.1M rows).

## 5. S1: the 10 floors, K1, K2 (`floors.py`, `chain.py`, `s1_floors.py`)

Every unconditional KN part and the KN-5 reference are tally `floor/kn.py` (sentence mode: one reply = one sentence
ending with `<end>`; modified KN, three discounts per order). Conditional levels are interpolated chains (`chain.py`):
"kn" levels use modified KN discounts from their own adjusted counts (tally's `discounts()`); "abs" levels use one
absolute discount per level D = n1 / (n1 + 2 n2) (0.5 if outside (0, 1)). A self-test asserts that the chain without a
condition equals tally's KN within 1e-12.
- F1 n {4, 6}: (S, w-n+1..w-1) -> ... -> (S) -> the unconditional KN-n (the condition stays at every level; the words
  back off first).
- F2 n {3, 5}: the per-condition KN chain (A, w-n+1..w-1) -> ... -> (A) -> uniform, mixed with the unconditional KN-n
  by one EM weight pair per frequency bucket of A.
- F3 a/b: absolute discounting, one level (R, w-1) [a] or (R, A, w-1) [b], R = the due multiset, backing off to F1 n 6.
- F4 eps {1e-2, 1e-4}: the tune-cal-better F3; placeholders with no due copy, and `<end>` while a placeholder is due,
  are multiplied by eps; renormalised.
- F5: absolute discounting (R, S, w-2, w-1) -> (R, A, w-1) -> (R, w-1) -> the tune-cal-best F2.
- F6: the tune-cal-best F2 and F4, mixed by one EM weight pair per frequency bucket of S.
- Buckets: the condition's count in the counting replies, {0}, {1}, {2-4}, {5-19}, {20-99}, {100-999}, {>= 1000};
  EM: tally `mix.em_bucketed`, tau 20 pseudo-tokens toward the global `mix.em` weights, fitted on tune-cal.
- Picks, all on tune-cal bits per token, ties to the earlier name: the better F2 and F3; then F4 and F5 on them; the
  better F4; F6; **the floor = the lowest of the 10**. tune-hold numbers of all 10 and of KN-5 are reported unchanged.
- **K1**: |PPL(tally KN-5) / PPL(KenLM `lmplz -o 5 -S 4G -T A/tmp/kenlm_sort`, `build_binary`, `query`) - 1| <=
  1e-3 on tune-hold (KenLM text: one reply per line, `<end>` dropped, `<unk>` renamed UNKPTB); the in-house KN-5 must
  have three count-estimated discounts at every order (no fallback): asserted. Otherwise STOP and fix (exit 3).
- **K2 (floors)**, before any number is logged: at 1,000 random tune positions (seed 0) every configuration's and
  KN-5's full vector sums to 1 within 1e-6, is > 0 and gives the scored bits (within 1e-9 bits).

## 6. S2: B, the learners, the read-out (`twins.py`, `learners.py`, `readout.py`, `s2b_*.py`)

- **Read-out, the same for every arm**: isotonic regression of the arm's score on the tune-cal rows (labels: the
  candidate is the true token; clipped out of range; q floored at 1e-6); P'(w) = (1 - b) P_f(w) + b P_f(C) q(w) / sum
  over C of q for w in C, P' = P_f outside C (and at a position whose true token is not in C); b in {0, 0.05, ...,
  0.95} by the tune-cal bits (ties: the smaller b). K2 for every configuration of every arm at the 1,000 positions.
- **Count twin** (8 configurations: keys {J2, J3, J4, J5} x w {3, 30}), counted on the twin rows: P(true | cell) =
  (positives + w P(parent)) / (rows + w), an unseen cell takes its parent's value. J1 = rank (9) x floor-probability
  bucket (8; log2 p thresholds -1, -2, -3, -4, -6, -8, -11) x type (5: due placeholder, other placeholder, end,
  punctuation, word) x due placeholders (0 / 1 / 2+); J2 = J1 x already said; J3 = J2 x top32 identity (33); J4 = J2
  x domain-act set; J5 = J2 x identity x domain-act set. Back-off: J5 -> J3; J3, J4 -> J2 -> J1 -> rank x bucket x
  type -> rank x bucket -> the overall rate. **B = floor + the tune-cal-best twin mix**, the best counting system
  (frozen in line R2).
- **Tree twin** per arm (8: min_samples_leaf {20, 100, 500, 2000} x alpha {1, 10}): sklearn DecisionTreeClassifier
  (entropy, random_state 0, no depth limit) on the arm's bits and the training rows; each node's P(true) = (positives
  + alpha P(parent)) / (rows + alpha) up the path.
- **LR1 / LR2** per arm (4 each: C {0.03, 0.3, 3, 30}): sklearn LogisticRegression (L2, lbfgs, max_iter 1000, tol
  1e-4); LR1 on the plain bits; LR2 adds every (position bit x candidate type flag or identity bit) product.
- **Boosted trees** per arm (8: max_leaf_nodes {31, 127} x learning_rate {0.05, 0.2} x l2 {0, 1}): sklearn
  HistGradientBoostingClassifier, max_iter 1000, early stopping (20 rounds, validation loss) on the training rows of
  the dialogues with sha256("es:" + dialogue_id) mod 10 == 0, which are not used for fitting; random_state 0.
- Picks: the lowest tune-cal bits per family and arm (ties: the earlier configuration); reported unchanged on
  tune-hold. Selection is optimistic on tune-cal (the picks, EM weights, isotonic fits and b are all fitted there);
  tune-hold is the unbiased check, and every gate is read on it.

## 7. CPU gates (tune-hold, per co-primary arm on its own bits)

Order, fixed (README job order): S0, bit-list line, S1, S2a, S2b-twin, **S2c (K8) before every learner**, the 8
learner jobs, S2d, then K3c for an arm that passed. S2c refuses to start once a learner folder exists; a learner
refuses to start before S2c is done or after a K8 STOP.
- **K8 (A/A, as amended in section 17)**: B rebuilt twice, each time with 5% of the fit dialogues dropped
  (`aa.subsample_train`) and the folds reassigned (`aa.folds`), once with seed 2 and once with seed 3; each rebuild
  recounts the S1 recipe and the twin configuration and refits the read-out. Rebuilt(2) minus rebuilt(3) on
  tune-hold must lie within +-0.0036 bits per token, with its 95% dialogue-bootstrap interval covering 0; else STOP.
  Both rebuilds count on 95% of the fit dialogues, so their difference holds only what an honest repeat varies.
  Reported, no rule (written after the K8 decision): B minus rebuilt(2) and B minus rebuilt(3), which also hold the
  effect of counting on 5% less data. The first freeze (05:13:40Z) registered B minus rebuilt(2) as the rule.
- **K9 (shuffled labels)**: the count twin, each learner's pick and the TM pilot retrained with the positive re-drawn
  uniformly among each position's candidates (seed 1), read out on the true tune-cal labels: chosen b = 0, or a
  tune-hold mix no more than 0.0010 below B; else STOP and find the leak.
- **K3 (i)**: the better of the boosted-trees mix and the LR2 mix at least 0.0439 bits per token below B (point
  estimates); **K3 (ii)**: the LR1 mix at least 0.0291 below B. 95% intervals printed beside them.
- **K3c**: our CoTM (dl/tm.py, workspace code, not a CAIR library) on the CPU (`astruct/tmrun.py`, G1's loop, torch at
  2 threads): 2 classes, 4,000 clauses, T 8,000, s {15, 40}, 10 epochs, batch 128, negation, hardest negative, init
  half, seed 1, on the training rows of the dialogues with sha256("pilot:" + dialogue_id) mod 10 == 0 (about 10%).
  The better pilot = the lower tune-cal bits (ties: s 15). Stop the arm if B minus that pilot's tune-hold mix is less
  than a third of B minus its boosted-trees mix, **or if B minus that pilot's mix is not above 0** (amendment 1,
  section 17: the pilot must be better than B). A pass needs the pilot's K9 at the better s. A real pilot starts
  only for an arm with `go_to_K3c` true, with exactly these settings, in its own folder (the shuffled one only at the
  better s and after a K3c pass); a real K3c decision refuses pilots with other settings (amendment 1).
- An arm that stops runs no GPU job; if both stop, the counting models are the result.

## 8. Later steps (code not written yet; it is hashed in a dated amendment line and a new freeze before it runs)

Rules from plan, section 3, unchanged: K4 (smoke), K4b (futility, T 8,000 x s {5, 15, 40, 100} first; best tune-cal
TM mix at least 0.0145 below B, else "FAIL at K4b (exploratory)"), the grid T {2,000, 8,000} x s {5, 15, 40, 100}
(4,000 clauses, 10 epochs, batch 128, negation, init half, G1's loop; score = class sum of "true" minus "not true"),
seeds 2-10 at the pick with their epoch-1 states, C0 x 10, the two untrained banks x 10, K5, K6, K7, line R2, the
final read, generation. Sub-choices fixed now:
- K4 smoke: one run on A-full's bits, T 8,000, s 15, seed 1, on the training rows of the dialogues with
  sha256("smoke:" + dialogue_id) mod 50 == 0 (about 2%). The one declared change: the s grid one step lower,
  {2, 5, 15, 40}, smoke at s 5; one retry; then stop.
- Grid picks: the lowest tune-cal mix bits (ties: lower T, then lower s); grid runs use seed 1.
- C0: the pick on the training rows of the dialogues with sha256("c0:" + dialogue_id) mod 100 == 0 (about 1%),
  seeds 1-10. Early checkpoint: each seed's own state after epoch 1.
- Untrained banks, draw d = 1..10 with numpy `default_rng(d)`; clause j gets as many plain and negated literals as the
  seed-1 TM's clause j (empty stays empty): uniform = drawn uniformly without replacement among the arm's features;
  data-anchored = plain literals from the 1-bits and negated from the 0-bits of a positive training row drawn
  uniformly (all available if too few). Clause outputs from `dl/cpu_tm_fast.py`; read-out v = ln((N_true + a) /
  (N_false + a)) over the training rows where the clause fires, score = sum over firing clauses, a on tune-cal from
  {0.01, 0.1, 1, 10}, then the common read-out. Vacuous = chosen b = 0 (reported with its firing rates).
- Final rows: built only by the frozen final script, with S0's rules and the full-fit floor and rates; their SHA-256
  logged; K2 asserted at every final position before any number is written.
- Generation (no rule): beam 5, at most 40 tokens, no length penalty; templates T1 / T2 as proposal A; BLEU-4 (NLTK
  `corpus_bleu`, delexicalised text, one reference) and SER = (missing + redundant placeholders) / expected.

## 9. Statistic, bars, wording (plan, section 3, unchanged)

- Metric: bits per token on the final split over every reply position (words, placeholders, `<end>`); perplexity and
  token counts beside it.
- Primary statistic: for every multi-seed arm (TM, C0, early checkpoint) and the untrained draws, per-position bits
  averaged over the 10 seeds or draws (bits, never probabilities); then a paired bootstrap over the 1,000 final
  dialogues, 10,000 resamples, seed 0. Two co-primary arms: every PASS interval is a 97.5% interval (Bonferroni).
  Reported beside each, no rule: a two-level bootstrap (seeds within the arm, then dialogues; 10,000; seed 0).
- PASS per arm (all four): (a) TM mix at least 0.0291 below B, upper bound of TM - B below 0, every seed's point
  estimate below B; (b) at least 0.0145 below each of C0 and the early checkpoint (10-seed means), both upper bounds
  below 0; (c) upper bound of TM minus the arm's tree twin below 0; (d) upper bounds of TM minus each untrained bank
  below -0.0072, and at least one bank non-vacuous. STRONG: PASS and the TM not above its trees mix or LR2 mix (seed
  mean). FAIL: anything else.
- Multiplicity: the two arms' PASS lines use 97.5% intervals; no other correction; each gate stands alone. The only
  claims: PASS and STRONG per arm.
- Duplicates: the share of final tokens inside a span of >= 8 tokens that also occurs in one fit reply, and every
  arm's result without them (the same is reported on tune-hold now).
- Wording fixed in advance (plan, section 3): x against the better of B and the arm's tree-twin mix; every result line
  names "our CoTM (dl/tm.py, workspace code, not a CAIR library)"; never "a TM writes language", never "the first";
  no comparison with published MultiWOZ BLEU or slot error; no tune number reported as a final result.

## 10. Power

The PASS margin is 0.0291 bits per token (2% perplexity); K3 needs 0.0439 and 0.0291 on tune-hold. S2d prints the
95% dialogue-bootstrap interval of every learner minus B on tune-hold (about 506 dialogues); line R2 states the
measured half-widths and the implied smallest gain the final split (1,000 dialogues) can show at 97.5%, before any
final read. The registered gates are the go-bars: no GPU job before K3 and K3c, no final read before K4b.

## 11. Run rules (enforced in code; README)

One Part 3 CPU job at a time across A and C (amendment 1: every real job starts through `chain/part3_cpu_job.sh`,
which holds `chain/part3_cpu.lock` for the whole job; `run_job.sh` refuses a real job unless that lock is named in
PART3_CHAIN_LOCK and held, and every real output needs nice 19, `A_RUNNER=1` from the runner's real mode and the held
lock); A's own flock; the CPU gate (`cpu_load.py 10`: at most 4.0
cores and 0 throttled periods, re-checked every 2 minutes, logged); nice 19; python -I -B; 2 threads (OMP, MKL,
OPENBLAS, NUMBA, RAYON; torch.set_num_threads(2)); sklearn without process pools; no GPU (CUDA_VISIBLE_DEVICES
empty); every cache and temporary folder and lmplz's -T in `A/tmp/`, lmplz -S 4G; the imported folders listed before
and after every job (exit 4 on any change); outputs never overwritten; the frozen-record check before every real
write.

## 12. Self-tests (run before the freeze and again on the frozen code)

`A/tests/run_tests.sh`: 63 pytest tests since amendment 1 (60 at the first freeze; counting venv) and 6 TM tests
(torch venv), then the listing check. Amendment 1 added or changed: K8 on the equal-size pair, recomputed from the
saved bits; K3c's pilot-beats-B clause on the review's case (trees worse than B, pilot worse than B: STOP); the real
pilot's and the real decision's refusals; the S-keyed lookup check with its negative control; the runner's real-mode
refusals (no chain lock, another lock, the chain lock as A's lock, the lock not held), the lock probe and the
test-mode mark; `check_runner` (nice, `A_RUNNER`, the chain lock). They
cover: the split rule; delexicalisation, overlaps and the token-count test (with negatives); conditions from
`dialog_act` only (span_info deleted; a leaky rule is caught); train-only reads and the manifest; the static check
that only S0 opens MultiWOZ and nothing names dev or test files; the vocabulary and `<unk>`; positions see only the
past; the chain equals tally KN without a condition (1e-12); all 10 floors normalised; the eps rule; candidates; the
frozen bit-list sizes at the real list lengths (364, 44 count-derived, 320); labels and flags; LR2 pairs against
brute force (and in chunks); shuffled labels; the review-2 leak test with its negative control; the rates' back-off;
the no-future test and an injected off-by-one leak it catches; count twin and tree twin against cell-wise
references; the oracle scorer reproduces the oracle bound (b = 1); a constant scorer gives b = 0, which equals the
floor; the sum contract catches a broken mix; gate arithmetic on made-up numbers (K1, K3, K3c with G1's 12.7%, K8,
K9, K4b, PASS incl. vacuous banks and one seed above B); the bootstrap equals tally's block bootstrap; the
frozen-record guard; the K7 clock; the bit-list LOG helper; the listing check, the gate parser and the runner (with
the real `cpu_load.py` line format); the TM loop learns a rule, equals `cpu_tm_fast`, is deterministic, needs no
CUDA, and the epoch hook sees every epoch; the dry run of every entry point in the job order on a synthetic corpus,
with every refusal (no bit-list line, a learner before K8, an A/A after a learner, the dry-run flags on a real output
folder) and every K3c branch (the trees' gain out of reach; a pilot worse than B; a pass, then the shuffled
pilot). The log of the run on the frozen code and its SHA-256 are named in the R1 line (for amendment 1, in its
amendment line).

## 13. What was known before R1, and what the known numbers predict

- No number of this pipeline on any MultiWOZ split. Train-only statistics: proposal A section 2 (56,581 replies with
  acts, 911,443 tokens by a simple regex, 2,066 word types with count >= 2, 61 placeholder types, 7,144 act-slot
  sets, 3,032 seen at least twice, 7.3% of a 12% hold-out with an unseen set) and the format check of the study log 03:55:36Z
  (35 act names, 29 slot names, 90,046 spans, 11 overlapping, 392 with a value that differs from the text, 1 span with
  an (act, slot) not in `dialog_act`).
- G1 (WikiText, another task; `glassbox-lm/studies/text_understanding/linec/g1_words.txt`
  879d4b0636edff21ce30ab8a19cf726ecadae828700543d2b550c0a680a615b2): floor 5.4147, count twin 5.3909, TM mean
  5.3828, LR 5.3678, boosted trees 5.3269 bits per token: the TM took 12.7% of the trees' gain over the twin.
- Synthetic dry run (240 tune-hold tokens; numbers are noise): every stage ran; K1's relative perplexity -1.6e-5.
- Predictions: K1 passes (A1 matched KenLM within 3.8e-5 on PTB; on the synthetic text the unigram count-of-counts
  that KenLM reports have one type fewer with adjusted count 1 than both tally and a hand count of its own input,
  a difference of about 2% in D1 there and negligible on 0.85M tokens). K2 and K9 pass. K3 (ii) is the likeliest stop
  (G1's LR was 0.0231 below its twin). K3c: G1's TM share (12.7%) would stop it. **K8 as first registered was at risk of a
  STOP that is not noise**: the rebuilt B counts on 5% less data, so B minus the rebuilt B is expected below 0 (count
  models gain roughly 0.1-0.3 bits per token per unit of ln(data), so a 5% cut costs about 0.005-0.015 bits); with
  about 506 tune-hold dialogues the interval is narrow, so it may exclude 0 or leave +-0.0036 although the margins are
  not inside the noise. My estimate of that STOP was about 0.6. **Amendment 1** registers the equal-size pair
  (rebuilt(2) minus rebuilt(3)) instead, whose expected difference is 0. The +-0.0036 clause should then pass easily.
  The interval clause can still fail when the difference between two rebuilds is larger than the dialogue
  bootstrap's interval for two near-identical systems; a STOP for that reason is accepted as registered. My guess of
  a K8 STOP now is about 0.15; it is not computed from any data. Overall priors (plan): about 0.04 for either
  arm (0.03 A-full, 0.015 A-nf).

## 14. Cost and time box

- The code review of 2026-10-21 measured 0.55-0.60 s per batch (2.5-2.7 h per pilot; up to 6 pilots, about 16 h).
- Measured on synthetic rows of the real width (364 bits, 4,000 clauses, T 8,000, 2 threads): 0.54-0.56 s per batch
  of 128, so about 2.5 h per K3c pilot (about 210,000 rows, 10 epochs); up to 3 pilots per arm (2 + the shuffled
  one), about 7.5 h per arm. The plan's step 5 estimate (1.2-3 h for all pilots) is too low; with S0-S2 (about 7-10 h,
  not yet measured) the CPU total stays under K5's 34 h.
- K5: the smoke run projects more than twice the plan's estimate (CPU over 34 h, GPU over 12 h): stop and re-plan.
- K7: no gate decision within 72 gate-open hours (`A/k7_clock.py` from the runner's log; a lower bound) or 10
  calendar days of starting S0: stop and report.

## 15. Implementation choices beyond the plan's text

The vocabulary order; the condition hashes; the overlap and range rules of the spans; the word-list tie rules; the
concept bits by spherical k-means (seeds 0 and 1); the thermometer thresholds of section 4; the candidate order; the
due-multiset hash R; the rate back-off chain; the KN-mode conditional chains and abs discounts; the EM buckets and tau;
the floor-probability buckets of the twin; F6's bucket on S (the plan's "act set"); refitting the EM weights of each
fold's floor on tune-cal; the early-stopping rows of the boosted trees; the salted row subsets (es, pilot, smoke, c0);
every seed above; K8 before the learners; the equal-size K8 and the reported B minus each rebuild (amendment 1); the
chain lock, the runner and settings checks and K3c's pilot-beats-B clause (amendment 1); the frozen-record guard; the sub-choices of
section 8.

## 16. Times and draft history

The code was written on 2026-10-21 from 02:56:20Z. Changes before the freeze, with no MultiWOZ number seen: the
frozen-record guard and `freeze.py`; K2 for all 10 floors before any number is logged and the K1 discount assertion;
LR2 pair features built in chunks and their partners found by name; `log_bitlist.sh`; `k7_clock.py`; the reported
second A/A rebuild; K8 moved before the learners with both refusals; a deterministic runner test. First draft of this
text: 2026-10-21T05:09:50Z. No rule changed after any number of this pipeline on a MultiWOZ split was seen, because none
exists.

Amendment 1 was written from 2026-10-21T05:59:18Z to its freeze, after the code review of A/ and C/ (2026-10-21). It is
listed in section 17. Its time and hashes are in the study-log line "R1 [A-Struct] AMENDMENT 1". The first freeze's
record, manifest, registration text and README are kept byte for byte in `A/history/` (`R1_A-Struct.frozen.v1.json` a2b7f6c8...,
`REGISTRATION_R1.v1.md` 2e8d13bb..., `MANIFEST_R1.v1.sha256` ec378fdf...; the full values are in the R1 line). At the
amendment `A/out/` was still empty: no S0-S2 job had run, and no MultiWOZ number of this pipeline existed.

## 17. Amendment 1 (2026-10-21, before any real run)

The code review of A/ and C/ (2026-10-21, verdict "ready after fixes") found no blocker, 2 major issues and 9 minor
ones (numbered 1-11 here in the review's order). Each change below was decided before any job of this pipeline ran on MultiWOZ, so no number could have steered
it. The section numbers name where the rule text now stands.

| # | Review issue | Before (first freeze) | After (amendment 1) | Where |
|---|---|---|---|---|
| 1 | major: K8 holds a data-size effect | K8 = B minus rebuilt(2): B counts on all fit dialogues, the rebuild on 95% | K8 = rebuilt(2) minus rebuilt(3), two rebuilds of equal size (each 95%, its own drop and folds); same bound (+-0.0036) and the same interval clause; B minus each rebuild reported with no rule | 7, 13; `s2c_aa.py`, `s2d_gates.py` |
| 2 | major: no lock across A and C | A locked only `A/tmp/cpu_job.lock`; C its own lock | every real job of A and C starts through `chain/part3_cpu_job.sh` (one flock on `chain/part3_cpu.lock` for the whole job, gate read inside it); `run_job.sh` refuses a real job without the held lock; the wrapper maps C's exit codes (lock busy 75, a gate STOP 3) | 11; `run_job.sh`, `lock_held.sh`, `astruct/paths.py`, `../chain/` |
| 3 | minor: K3c passes a pilot worse than B when the trees are worse than B | stop if B - pilot < (B - trees) / 3 | also stop unless B - pilot > 0 | 7; `astruct/gates.py` |
| 4 | minor: pilot and decision guards | a real pilot could run for a stopped arm or with the dry-run settings; the decision read any pilot | a real pilot needs go_to_K3c[arm], the registered settings and its own folder (the shuffled one the better s and a K3c pass); a real decision reads only `A/out/` and refuses pilots with other settings | 7; `k3c_pilot.py`, `k3c_gate.py` |
| 5 | minor: scripts run directly skip the runner | `check_env` only | a real output also needs nice 19, `A_RUNNER=1` (the runner's real mode) and the held chain lock | 11; `astruct/paths.py` |
| 9 | minor (optional): fold floors' S-keyed levels not asserted | rate_set only | also every S-keyed context lookup of the fold floor unseen, and S's count 0, for singleton act-slot sets | 4; `astruct/build.py`, `s2a_rows.py` |
| 11 | minor: time boxes | 0.54-0.56 s per batch, about 2.5 h per pilot | the review's 0.55-0.60 s (2.5-2.7 h; up to 6 pilots, about 16 h) added; K7's clock stays a lower bound | 14; README |

- Not changed, with reasons. The review's issues 7, 8 and 10 concern C only; C's code is unchanged, and its next
  amendment carries them. For issue 6 (C's ".." listing lines), the chain wrapper treats a change confined to ".."
  lines as a note; A's `listing.py` already did. K8's interval clause is kept: the review offered dropping it, but
  keeping it changes the plan's rule least. The equal-size pair removes the data-size effect that would have made
  it fail; the risk that remains is stated in section 13.
- Deviations from the plan's text, declared here: K8 now compares two rebuilds instead of B and one rebuild, and K3c
  has one more stop clause (the Part 3 plan, section 3, "Kill criteria" and "Gates before any GPU job"). The K3c clause
  is stricter. The K8 change is not stricter: it removes a stop that would come from the 5% data cut rather than
  from noise, and keeps both of the plan's clauses for the noise that K8 exists to measure. It was decided with no
  number of this pipeline known. The plan is not edited, because its SHA-256 is an input of R1.
- Unchanged: every other rule, grid, seed, split, margin and wording of sections 1-16 and of the plan.
- Tests: `A/tests/run_tests.sh` and `chain/tests/test_part3_cpu_job.py` were run on the amended code; their
  results, logs and SHA-256 values are in the amendment line of the study log.
