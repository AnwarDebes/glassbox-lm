# PLAN: text understanding for the Tsetlin machine and the agent (main line from 2026-10-19 21:34Z)

Status at writing: plan only, nothing run. The goal of the line (2026-10-19 21:34Z), in the author's words: "our
biggest problem in this project is to make the tsetlin machine understand text, also make our agent understand text,
which we call language embedding or text embedding. So this is our main focus since without it, we would not be able
to go very far."

Every step below gets a registration line in the study log, with code hashes and bars, before it runs. Validation splits
only; no test split is read without its own registration. GPU work comes after N9 and decision_layer's R24.

## 0. Binding rules for every step

From the project's map of the text stack (2026-10-19 21:46Z). The goal, in the author's words: "find a solution for every problem that prevents the Tsetlin machine from text embedding, also learning
languages. All the solutions should work together, not a solution that destroys another." Every step below obeys:

- **R1, one layer each.** The layers are L0 tokens, L1 word table, L2 bit blocks, L3 reader, L4 users.
  - Bit blocks are ADDITIVE: a new block never removes existing bits.
  - Tables are swappable, but the C4 counting table stays the reference with no neural network at any stage. It is
    reported beside any teacher table.
- **R2, stacking.**
  - Each candidate is tested ON TOP OF the current best stack, not alone, with the same bar: +0.5 with both probes
    (TM and LR) on 2 of 3 intent val sets.
  - At each milestone, an ablation (the stack minus each piece) checks that every piece still pays. A piece that no
    longer pays in the stack is dropped, with the reason logged, even if it helped alone.
- **R3, re-tune the reader** (s by the e74 rule, the same values for every view) after each accepted change.
- **R4, no regression in the agent.**
  - Every accepted stack also runs in the decision_layer agent on STAR and ABCD dev at its tuned s.
  - A stack that costs the agent more than 0.5 points on dev is not taken into the agent. It may still be kept for
    classification, labelled so.
- **R5, the guards stay provable.** A new block enters the guard domain (e70 / e72d level-B rows), and the 50 guard
  proofs are re-run for the new view before the agent uses it.
- **R6, provenance on every line.** Either "no neural network at any stage" (counting tables, logic blocks), or
  "network-free at run time, teacher at training time" (3a, 3b, 3c).
- **R7, test splits only at milestones**, through a registration frozen first, with the use count declared.
- **R8, keep everything.** Nothing in the current work is replaced. The agent results, R24, the guards and word-edit
  proofs, the harness plan, N9 and the e71 / e76 runs all continue.

**Order of work:**
1. Step 1 (running).
2. Step 0b after N9: the new base stack.
3. Steps 2a / 2b / 2c / 2d, in the order step 1's shares give, each on top of the stack.
4. Then 3a, 3c and 3b.
5. Milestone: the agent check (R4), the guards (R5), then one registered test (R7).
6. Later: language generation on the better stack (P10) and a Norwegian table (P11).

**Two problems this plan adds:**
- **P7, saturation of OR bits on long texts** (step 2d):
  - First measure the on-share of existsT bits by text length: on intent val here, and on the agent's long customer
    texts (STAR, up to 548 tokens) in the decision_layer study.
  - If the on-share rises toward saturation with length (to be defined from the measurement and fixed before any
    run), add a block of concept counts or positions: per concept, a thermometer of how many tokens hit it, or bits
    for the first or last third of the text.
  - Same bar, on top of the stack.
- **P7 result and the 2d block definition (2026-10-19 22:06Z-22:08Z; plan only; run in the decision_layer agent):**
  - **Intents:** the OR bits fill with length (Spearman +0.86 to +0.91). The longest 10% of texts reach 0.70 / 0.78 /
    0.68 on-share (CLINC150 / Banking77 / HWU64, C4 view). Only Banking77 is over the 0.75 line.
  - **The agent's texts** (c4_saturation.json 8a368035, C4 view, dev): STAR xcust 0.815 and ABCD xcust 0.767 are
    SATURATED. STAR xrecent 0.74 and ABCD xrecent 0.72 are not.
    - Texts with 90% or more of their bits on are rare (0.10% / 0.03%), so the saturation is soft.
    - C4 and potion give the same curves.
  - **So 2d is tested in the agent, for xcust only.** The block is ADDITIVE beside xcust (R1) and never replaces it.
    Two candidates under one rule, in this order:
    - **2d-a, concept counts for the customer's words so far.**
      - For each concept k: n_k = the number of the customer's tokens so far whose cosine to k's centre is at or
        above k's lowest thermometer level (the level xcust's first bit for k uses). Thresholds come from the fit on
        the training rows.
      - Bits: n_k >= 2, n_k >= 4, n_k >= 8, so 3 per concept and 1,536 in all. (n_k >= 1 is already xcust's first
        bit, so it is not repeated.)
      - What it adds: the frequency that the OR throws away ("mentioned once" vs "the topic of the conversation").
      - It is a table lookup and a sum: no neural network at any stage on the C4 table.
    - **AMENDMENT to 2d-a (2026-10-20 00:24Z; decided before any 2d-a run, from bit statistics only, no model).**
      - Why: counts at xcust's LOWEST level saturate in long texts (measured in the decision_layer study). On STAR's longest 10% the bits
        n >= 2 / 4 / 8 are on at 1.00 / 0.99 / 0.91, so the block would carry nothing exactly where it should help.
      - At level 3 (the median level of k's thermometer), the longest 10% give 0.66 / 0.34 / 0.07 on STAR and
        0.52 / 0.20 / 0.02 on ABCD.
      - 487 (STAR) and 474 (ABCD) of 512 concepts have a level 3.
      - The amended block: n_k counts the customer's tokens so far whose cosine to k's centre is at or above k's
        LEVEL-3 threshold. Where k has no level 3, its highest level is used. The bits stay n_k >= 2 / 4 / 8.
      - Amended R5 domain rows: n >= 8 implies n >= 4 implies n >= 2 implies xcust's level-3 bit for k (or its
        highest-level bit where k has no level 3).
      - Amended self-test: n >= 1 at level 3 equals xcust's level-3 bit (or highest-level bit) for every concept,
        row for row.
    - **2d-b, the customer's recent utterances: "view B + xcrecent".**
      - The existing part xcrecent of ABCDView (R20): existsT bits of the customer's last 3 utterances only
        (dl.abcd.customer_recent), added on top of view B. It needs NO new code and is tested code. It was used in
        R20's view C, which had no human-agent text, but never on top of B.
      - It replaces a first draft (the latest utterance only, n = 1). That draft needed a new subclass, and the
        existing n = 3 part is already tested (noted 22:07Z).
      - What it adds: a shorter, unsaturated customer-only signal beside the cumulative xcust.
      - It overlaps partly with xrecent (the last 3 turns of both speakers, already in view B), so it is second.
  - **The rule:**
    - Each candidate is tested on top of the current agent stack (R2), at the stack's tuned s.
    - 2d-a first. 2d-b next, on top of whatever is accepted by then.
    - Bar (fixed 22:06Z): STAR dev at least +0.5 and ABCD dev no worse than -0.5, mean of seeds
      1-2.
    - On acceptance: R3 re-tuning of s, then R5. The new bits enter the guard domain: domain rows say that the
      count thermometer chains imply each other (n >= 8 implies n >= 4 implies n >= 2) and that n >= 2 implies
      xcust's first bit for k. Then the 50 proofs are re-run before the agent uses the block.
  - **Priors:** 2d-a about 25% (the saturation is soft); 2d-b about 15% (xrecent already carries much of it).
- **P11, other languages:** a Norwegian table counted from Norwegian text by the same pipeline (word list, counting +
  SVD), then existsT. No new method, later. The Norwegian R12 result used a transformer.

## 1. What we know (numbers in the study log; intent val; TM = mean of seeds 1-3 unless marked)

| Text bits read by the TM (existsT unless named) | CLINC150 | Banking77 | HWU64 |
|---|---|---|---|
| words only, bag of words (test 1, seed 1) | 81.23 | 80.36 | 77.42 |
| C4 counting table ppw_S1000 (no neural network at any stage) | 92.29 | 87.91 | 89.34 |
| potion-base-32M (a static table distilled from bge-base) | 94.60 | 91.46 | 91.23 |
| MiniLM sentence-embedding bits (test 1, seed 1; the study's rows) | 94.57 | 91.82 | 90.24 |

- **The gap that matters** for a network-free run time is C4 table to potion: 2.31 / 3.55 / 1.89 TM points.
  - With the TM as the reader, potion's existsT is already level with MiniLM's sentence bits (seed-1 rows).
  - Logistic regression on MiniLM bits is higher, for example 95.77 on CLINC150. So part of the remaining ceiling
    is the TM, not the bits.
- **On topic benchmarks the C4 table is on par with potion** (W1-T, registered, N7's test splits). The agent with no
  neural network at any stage is non-inferior to a bge-large agent on STAR test (R22, +1.64; not "beats").
- **D0's diagnosis:** the validation examples only potion gets right are mostly common words (87% / 98% / 94% contain
  no word missing from the C4 table). The misses are meaning:
  - negation ("i dont want sound" -> mute);
  - antonyms (deactivate / activate);
  - paraphrase ("what's the clock say" -> time; "idk" -> maybe);
  - brand names, and typos (CLINC150).
- **What our bits cannot see:** existsT is a bag of concepts with OR. It cannot see word order or negation scope.
- **What failed:**
  - a TM autoencoder making the word table (W1; counting beat it);
  - question->answer pairs from C4 (W2 H-B; no gain);
  - a hand-curated lexicon (W3; counter-fitting made the table worse).
- **What is running:** N9 tests whether the meaning bits add over bag of words once the combined view is tuned like
  bow (TREC, MR).
- **Not tried yet:**
  - negation-scope bits;
  - ordered concept pairs or windows;
  - teacher distillation into a counting table or into TM bits;
  - paraphrase augmentation;
  - a character n-gram table for typos and rare words.

## 1b. Step 0b: the TM reader's own setting (added 21:38Z)

- **Why:** every intent-set TM so far ran at test 2's setting (4,000 clauses, T 4000, s 8). Logistic regression on
  MiniLM bits beats the TM (95.77 vs 94.57 on CLINC150), so part of the ceiling is the reader.
- **Evidence that s matters:** on STAR, tuning s alone gave the agent +2.70 on test (R22). In N8, tuning moved TREC
  +8.3 and MR +3.4.
- **The screen:** s in {2, 4, 8, 12, 16, 24} at 4,000 clauses and T 4000, on intent val, seed 1, for BOTH the C4-table
  view (ppw_S1000 existsT) and the potion view. Everything else stays at the harness's setting.
- **The pick, e74's rule:**
  - per set and view, the s with the highest val accuracy;
  - if that is less than 0.3 points above s 8, the pick is s 8;
  - an exact tie goes to the s listed first.
- **Numbers:** the picks run on fresh seeds 4-6, the same seeds for both views. s 8 also runs on seeds 4-6 as the
  reference.
- **What is reported:** the C4-potion gap at equal tuning (picks, seeds 4-6) against the gap at s 8 (seeds 4-6), and
  the share of the gap that tuning closes.
- **Bars, fixed now:**
  - "tuning helps a view" on a set: pick minus s 8 at least +0.5 (mean of seeds 4-6, all 3 paired seeds positive);
  - "tuning closes the gap" on a set: the C4-potion gap at the picks is at most half the gap at s 8.
- **What follows:** steps 2 and 3 then run at the picked s of their view (declared). If tuning closes most of the gap,
  that is the cheapest win of the plan and sets how much is left for steps 2-3.
- **Cost:** 6 values x 2 views x 3 sets = 36 screening runs of about 30-45 s, plus up to 2 x 2 x 3 x 3 = 36 seed runs.
  That is about 1 GPU-hour, after N9.
- **Prior:** "helps" on at least one view and set about 50%; "closes the gap" on at least 2 of 3 sets about 15%.
  Tuning will likely lift both views.

## 2. Step 1: a precise diagnosis (CPU only, minutes; no bar, decision rules fixed first)

**Probes.** W2's logistic-regression probe (dl.linear on the CPU, as in D0) on intent val, on three bit sets:
- ppw_S1000 existsT;
- potion existsT;
- MiniLM thermometer bits (the decision-layer study's cached MiniLM embeddings, data/emb/{ds}__minilm.npz).

**Groups.** potion-only vs ppw wins, MiniLM-only vs potion wins, and MiniLM-only vs ppw wins.

**Types, by automatic rules fixed before the run:**
- N, negation: a negator token (not, no, never, n't, dont, cant, cannot, without, nothing, none) is present.
- R, rare word: a token with fewer than 5 occurrences in that dataset's training split, or no C4 row.
- T, typo: a token with no C4 row that is within edit distance 2 of a C4 word.
- P, paraphrase: none of the above, and content-word Jaccard below 0.5 to the nearest same-class training example.
- Order: the rules cannot see it. The hand check below labels it.

**Hand check.** A stratified sample of 100 potion-only and 100 MiniLM-only examples, each given one main type (N, O
order, A antonym, P, R, T, B brand / entity, X other). Judged by an AI model, not a person. Wilson intervals on the
shares.

**Decision rules, fixed now:**
- N plus A at least 15% of the wins: run step 2a.
- O at least 15%: run step 2b.
- R plus T at least 20%: run step 2c.
- P at least 50%: the teacher route (step 3) is the main bet. It runs whatever step 1 says.

**Cost.** CPU about 15 minutes, plus the hand check.

## 3. Step 2: logic features a TM is good at (small val tests, both probes, 3 seeds)

Every step-2 view is the old existsT bits plus the new bits. The bar for each is the one of W2 and W3: at least +0.5
over existsT alone with BOTH probes (TM and LR) on at least 2 of the 3 intent sets. Each step also has a target-
subset readout: accuracy on the val examples of its own type from step 1. That readout is descriptive.

**2a. Negation scope.**
- A token is "under negation" when one of the negators occurs within the 3 tokens before it, with no punctuation in
  between. This is the NOT_ marking of Das and Chen 2001 and Pang et al. 2002 at the concept level.
- New bits: per concept k, one bit "a token under negation has its top concept k with cosine at or above k's lowest
  thermometer level". That is 512 bits.
- Antonyms get no special bits; step 3 handles them.
- Prior: about 15%. Negators occur in only a few percent of intent texts, so the possible gain is small.
- GPU: about 20 minutes.

**2b. Order.**
- Ordered concept pairs: for the 4,000 most frequent ordered pairs (a before b, within 4 tokens) in the training
  split, one bit each.
- A variant to consider only if 2b passes: a convolutional TM over the token sequence (tmu's patch convolution on a
  1-D token "image"). That needs new code and its own rule.
- Prior: about 10%.
- GPU: about 30 minutes.

**2c. Typos and rare words** (only if step 1 says so).
- A character 3-5-gram counting table over C4 (PPMI-SVD of character n-grams by context words). A word's vector is
  the mean of its n-gram vectors, used where ppw_S1000 has no row.
- Still no neural network at any stage.
- Prior: about 20% on CLINC150 only.
- CPU about 1-2 hours, GPU about 20 minutes.

## 4. Step 3: the teacher route (network-free at run time; a teacher network at training time)

Said plainly in every report: these students learn from a neural teacher at training time, as potion does (Model2Vec
distils bge-base). At run time they need no neural network.

**3a. A counting table refined by a teacher (Tokenlearn-style, linear).**
- Start from ppw_S1000's rows.
- Fit the word vectors so that the weighted mean of a C4 sentence's word vectors matches the teacher's sentence
  embedding, by least squares. Teacher: MiniLM, all-MiniLM-L6-v2, Apache 2.0, already cached in the study. Data: 2M
  C4 sentences from S1000, never evaluation text.
- Then existsT and the TM, as always.
- This is potion's recipe on our C4 word list. A table distilled from MiniLM is potion's kind of table: say
  "network-free at run time", never "no neural network at any stage". The C4 counting table stays the reference
  with no neural network at any stage.
- If it reaches potion, it is a replication, not a new method. Its value is a table we build and inspect end to end.
- Bar: +0.5 over ppw_S1000 with both probes on 2 of 3 sets. The share of the gap to potion is descriptive.
- Prior: about 40%.
- GPU: teacher encodings, about 30 minutes for 2M sentences. CPU: the least-squares fit.

**3b. A TM student of the teacher's bits.**
- A multi-output TM (w1/mo_tm.py, already written for W1) reads the token-concept bits of a C4 sentence. Inputs are
  existsT from ppw_S1000, plus 2a's and 2b's bits if they passed.
- It outputs the teacher's thermometer bits: MiniLM 384 dimensions x Thermo(4) = 1,536 outputs.
- At run time the classifier reads the predicted bits beside existsT.
- Twins on the same inputs and targets: a linear student (one logistic regression per output bit), and a
  counting-only student (per-concept averages).
- Bar: the TM student's predicted bits plus existsT beat existsT alone by +0.5 with both probes on 2 of 3 sets, and
  beat the linear student's predicted bits.
- Prior: about 10-15%. W1's multi-output TM lost to counting on a similar job.
- GPU: about 3-4 hours.

**3c. Paraphrase augmentation with the local small LLM** (run in its own venv).
- 5 label-preserving paraphrases of each training text, written by the local LLM from a fixed prompt.
- Training texts only. Validation and test texts are never shown to the LLM.
- Kept when the teacher's cosine to the original is at least 0.7 (fixed now).
- The TM classifier on ppw_S1000 existsT is trained on originals plus paraphrases.
- Bar: +0.5 with both probes on 2 of 3 sets.
- Prior: about 30%. Augmentation helps most with little data; our training sets are full size.
- GPU: about 1-2 hours of generation, depending on the model and the engine.

**Order of step 3:** 3a first (cheapest, best prior), then 3c, then 3b.

## 5. What the agent gains

The decision-layer agent reads the same kind of text bits (word table, existsT, thermometer). A step that passes on
intents is carried to the agent's dev sets (STAR, ABCD) with its own frozen plan. The agent's current text side is
already non-inferior to bge-large on STAR test (R22). So intents are where text understanding is still weak.

## 6. Novelty, newest work and current baselines (novelty/text_understanding_prior_art.md, 05dbaf51d8c7b7e0; searched 2026-10-19)

**What is old:**
- Negation-scope marking: NOT_, 2001-2002. It did nothing on long reviews, and added +1.8 to +3.1 F on tweets, where
  13.5% of texts hold a negation.
- Bigrams: +0.2 to +1.8 on short sentiment texts, about 0 on topic sets.
- Role bits for questions: +5.4 on TREC (Huang 2008).
- Static students of a sentence encoder keep 87-95% of the teacher on MTEB classification: Wada et al., EMNLP 2025;
  potion against bge-base, 95%.
- LLM paraphrase augmentation with full training data: +0.3 to +0.7 (RoBERTa-large, Lin et al., EACL 2023), and on
  Banking77 +0.4 unfiltered or +1.1 filtered (BERT-base, Findings of EMNLP 2025). Unfiltered LLM data hurt Banking77
  and HWU64 in two papers, because the generator wrote examples of a neighbouring intent.

**What would be new, narrowly:** we found no published study that:
- gives a TM negation-scope bits ("negation" in TM papers means negated literals);
- distils a neural sentence encoder into TM-readable bits (TM-to-TM distillation gave +0.8 on IMDB, Kinateder 2025,
  who names network-to-TM as future work);
- tests negation, order or LLM augmentation for a TM, bag-of-words or static-table classifier on CLINC150, Banking77
  or HWU64.

**Newest work** (searched 2026-10-19):
- **TM work:** the ISTM 2025 and 2026 programmes and arXiv up to 2026-10-09 hold no TM work on negation, word order or
  intents.
- **Teacher-scored n-grams** are the one proven way to add order to a static student. Aug-GAM beat the same teacher's
  word vectors by +2.3 to +6.4; Luxical (December 2025) is an n-gram student of a transformer. A possible 3a variant:
  teacher-refined vectors for the frequent bigrams as well, fixed in 3a's registration if used.
- **The strongest open teacher found:** Qwen3-Embedding-8B, MTEB(eng, v2) classification 90.43 (June 2025). Too large
  and fp16-risky for our V100.
- **Practical teachers:**
  - all-MiniLM-L6-v2 (Apache 2.0), already cached;
  - bge-base-en-v1.5 (MIT), potion's own teacher, a download to be checked first. With it, a 3a table would
    be compared with potion under the same teacher.
  - The teacher is chosen in 3a's registration.

**Current state of the art on the three sets, full training data** (test numbers; our numbers are val, so placement
only):

| Set | Best published | Older static / bag-of-words rows | Our C4 pipeline, TM val |
|---|---|---|---|
| CLINC150 | 97.80 (SPACE-2, 2022); 98.45 arXiv only (TOATOD, 2023) | 89.0-91.5 (2019) | 92.29 |
| Banking77 | 94.77 (SPACE-2, 2022) | none published | 87.91 |
| HWU64 | 94.33 (SPACE-2, 2022) | none published | 89.34 |

- All the best numbers come from large pretrained transformers, 5-7 points above our network-free pipeline.
- We found no 2024-2026 paper with a static or bag-of-words classifier on these full sets.
- **Expected sizes from the literature:**
  - negation or order bits: +0.2 to +1.0 per set (not run; step 1's shares rule them out);
  - distillation (3a): +0.5 to +2 at best, up to potion's level unless n-grams add context;
  - filtered paraphrases (3c): +0.3 to +1.5.
  - This matches the priors of section 8.
- **Wording rules:**
  - "we found no published study that ..."; never "first";
  - cite Model2Vec, Tokenlearn and Wada et al. for 3a; Das and Chen and Pang et al. for 2a; Lin et al. 2023 and the
    Findings of EMNLP 2025 Banking77 study for 3c.

## 7. Budget and order

1. Step 1: CPU only; it can run during N9. Step 0b: about 1 GPU-hour after N9.
2. Step 2: about 1 GPU-hour in all.
3. 3a: about 0.5 GPU-hour plus CPU.
4. 3c: about 1-2 GPU-hours, after the local LLM's smoke test.
5. 3b: about 3-4 GPU-hours.

Total about 6-8 GPU-hours, all after N9 and R24.

## 8. Honest priors in one place

| Step | Prior |
|---|---|
| 2a negation | about 15% |
| 2b order | about 10% |
| 2c typos | about 20%, CLINC150 only |
| 3a teacher-refined counting table | about 40% |
| 3b TM student | about 10-15% |
| 3c paraphrase augmentation | about 30% |

- Each step reaching potion's level (closing the 2-3.5 point gap) is less likely than its bar: about 20% for 3a, under
  10% for the others.
- The most likely good outcome is 3a: a C4 table refined by a teacher at training time, reaching or nearing potion.
  It would be network-free at run time, and it would be a replication of potion's recipe, not a new idea.
