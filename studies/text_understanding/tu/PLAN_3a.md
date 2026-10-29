# Step 3a: the C4 word table refined by a teacher (plan for registration)

Step 3a of the text-understanding plan. The teacher is bge-base-en-v1.5,
not MiniLM (decided on 2026-10-19 at 22:11Z), so that the result compares with potion under potion's
own teacher. Nothing in this file has run yet.

## Question

Can the C4 counting table (ppw_S1000) close its intent gap to potion when it is refined toward a teacher's sentence
embeddings? The refinement is potion's recipe (Model2Vec / Tokenlearn): least squares under mean pooling, applied to
our own C4 word list.

**How to say it, in every report:** the refined table learns from a neural teacher at training time, as potion does.
At run time it is a lookup table, so the reader is "network-free at run time". Never write "no neural network at any
stage" for it. The C4 counting table stays the reference with no neural network at any stage. If 3a reaches potion,
it is a replication of potion's recipe on our word list, not a new method.

## Inputs

- **Start table:** ppw_S1000_d500_e0.5 (w1/tables/ppw_S1000_d500_e0.5.npz, sha256 e1771d9eaf4a07f7): 63,091 C4
  words x 500 dimensions, unit rows.
- **Teacher:** bge-base-en-v1.5 at revision a5beb1e3 (MIT; data_untrusted/bge-base-en-v1.5/a5beb1e3/; every file
  checked against FILES.sha256).
  - It is loaded with transformers' BertModel from the local safetensors only, with no remote code.
  - Sentence embedding as the model card defines it: the [CLS] vector of the last layer, L2-normalized, with no
    instruction prefix.
  - fp16 on the V100. At most 128 tokens per sentence; the share that is truncated is reported.
- **Text:** 2,000,000 training sentences and 40,000 held-out sentences from W1's C4 S1000 documents.
  - These are W1-S's stream order, cleaning and evaluation-text filter: the key rule and the 8-gram rule against
    every evaluation split, including ABCD and STAR dev and test. So no evaluation text gets in.
  - Pool: the first 2,243,398 documents of that stream, which is S1000's document count. No potion tokenizer is used
    here, so the pool can differ from S1000's end by the few documents W1 cut for an empty potion encoding. Declared.
  - Sentences are split by W1's rule. A sentence is eligible when it has 3-64 words by our tokenizer
    (glove_feats.glove_tokens) and at least 3 of them are in the 63,091-word list.
  - Sampling: documents in a numpy default_rng(20261020) permutation. Each eligible sentence is kept with
    probability 0.1 (the same generator) until 2,040,000 are kept. The first 40,000 kept form the held-out set; the
    rest form the training set.

## The fit (linear, closed form plus conjugate gradients)

- **Student pooling:** a sentence's vector is the mean of the rows of its known words, with repeats counted. This is
  a sparse matrix A (sentences x 63,091), with a_nw = count of w in n / known words in n.
- **Targets:** the teacher vectors, centred by the mean over the training sentences. The held-out set uses the same
  mean. No PCA; 768 dimensions.
- **Table a3A** (descriptive): one linear map M (500 x 768) from ppw_S1000's pooled sentence vectors to the targets,
  by ridge least squares (mu = 1e-6 x trace / 500). Rows = ppw_S1000's rows x M.
- **Table a3B** (the registered arm): every word row is free.
  - It solves (A^T A + lambda I) W = A^T T + lambda W_A by conjugate gradients with the Jacobi preconditioner. The
    lambdas run in the order 10, 1, 0.1, 0.01. The first starts at W_A (a3A's rows before normalising); each next one
    starts at the previous solution (the same solution at convergence, in fewer steps). It stops at a relative
    residual (Frobenius) of at most 1e-4, or after 400 steps.
  - All four lambda tables are kept (a3B_lam*.npz). Only lambda*'s table is probed.
  - A word that never occurs in the training sentences keeps its a3A row. A word whose ppw_S1000 row is zero (dead)
    stays dead in both tables, so the vocabulary is exactly ppw_S1000's.
  - lambda is chosen from {0.01, 0.1, 1, 10} by the highest mean cosine between student and teacher on the
    HELD-OUT sentences. Ties go to the larger lambda. The intent sets play no part in this choice.
- Both tables are written with unit rows in ppw_S1000's format (words, codes float16, counts, vocab_ids), to
  tu/a3/tables/.
- **Reported:** held-out cosine for ppw_S1000 through M (= a3A) and for a3B at each lambda; CG iterations; the share
  of words that occur at least once; truncation.

## Probes (val only; the test split is never read)

- **Features:** w1/wordtab_feats.py, unchanged (001a93b259a1b34f): K 512, Thermo(8), existsT. One build per table and
  set: CLINC150, Banking77, HWU64.
- **TM:** tu/a3_exists_tm.py = tu/b0_exists_tm.py with only its output files and study label changed. It runs at the
  C4 view's s picked by 0b, on seeds 4-6.
  - Reference: 0b's ppw_S1000 rows at the same s and seeds, paired by seed.
- **LR:** tu/a3_lr.py = w3/w3_lr.py with only its output file changed (C in 1/10/100, best on val).
  - Reference: ppw_S1000's LR rows in w2/results_w2_lr.jsonl, from the same code path.

## Bar (fixed now; from PLAN_text_understanding.md section 4)

- **PASS:** a3B minus ppw_S1000 is at least +0.5 with BOTH probes (TM mean of seeds 4-6; LR val) on at least 2 of the
  3 sets.
- **Otherwise FAIL.**
- **Descriptive only:**
  - the share of the C4-to-potion gap that is closed (TM: potion at its own 0b pick, seeds 4-6; LR: potion's LR row);
  - a3A's numbers;
  - the paired TM seeds.

## Tests before the GPU job

- **The sampler**, on CPU: it is run for real (data only, no result). Its counts, its filter drops and the sha256 of
  the sentence files go in the study log.
- **The fit:** a synthetic self-test with a planted table, where CG must recover it within 1e-3 at a small lambda.
  - Also: the step-A map on synthetic data equals numpy's lstsq, and the held-out cosine code matches a direct loop.
- **The teacher:**
  - one CPU check on 64 sentences, that fp32 on CPU and fp16 on GPU agree within 1e-2 in cosine (the GPU part runs
    inside the GPU job, before the full encoding);
  - a sanity check that 3 paraphrase pairs score higher than 3 unrelated pairs. It is reported, not a bar.

## GPU job and order (after 0b, at its turn in the queue)

1. Teacher encoding: about 10-15 minutes.
2. Fit: about 10-20 minutes.
3. 6 feature builds, 18 TM runs (2 tables x 3 sets x 3 seeds) and 6 LR fits: about 30 minutes.

That is about 1 GPU-hour in all. One big GPU job at a time; the CPU gate is cpu_load.py at most 4.5 cores.

## Prior

About 40% (plan). A single-table replication of potion's recipe has a good chance to move CLINC150 and HWU64. It is
less sure for Banking77, whose gap is the largest (3.55).

## Amendments after a pre-run review (2026-10-20 19:20Z; all before any 3a GPU step)

1. **MultiWOZ filter (leakage).** W1's evaluation filter predates MultiWOZ 2.2, which is now an evaluation set of the
   agent study (e77-e79, R24).
   - The sampled sentences are filtered after sampling by the same key and 8-gram rules against the MultiWOZ 2.2 dev
     and test utterances. These are the "utterance" text of every turn, read as mwoz_leak/mwoz_c4_overlap.py reads
     them. The script is tu/a3_mwoz_filter.py.
   - Held-out set: 40,000 -> 39,995 (3 dropped by key, 2 by 8-gram).
   - Training set: 2,000,000 -> 1,999,932 (29 by key, 39 by 8-gram).
   - No resampling; the counts are declared as they are.
   - The teacher and the fit read the filtered files c4_sent_{held,train}_f.txt.
2. **Verdict rounding.** The bar compares the unrounded gains with 0.5 - 1e-9. The same fix goes into 0b's verdict:
   its gaps are now taken from the unrounded means.
3. **LR reference.** The same job runs a3_lr.py on ppw_S1000's own feature files. That fresh row is the LR reference;
   the W2 row is reported beside it.

## Notes (review of 19:20Z)

- bge-base's training data cannot be checked for these intent sets. The same holds for potion's teacher, so the
  comparison is fair. Declared.
- "Registered" here means pre-registered on val. No test split is used.
- On PASS, the a3B table is exported to the decision_layer study, the way c4words_S1000_v1 was, for R4: the agent on STAR, ABCD
  and MultiWOZ dev. R5 (the guards) comes after that.

## Post-run notes (2026-10-21, after the verdict; nothing above is changed)

From three read-only checks (leakage, selection, integrity; judged by an AI model, not a person), all of which
CONFIRMED the PASS
(pre-registered val PASS):
- **LR reference:** the "Probes" section and the 19:20Z registration still name the W2 LR row as the reference.
  Amendment 3 replaced it with a fresh ppw_S1000 row from the same job, and its values equal the W2 rows exactly.
- **lambda:** lambda* = 0.01 lies at the edge of the grid, and the held-out cosine was still rising there.
  - Ranked on the unit-row tables, the same criterion puts 0.1 just above 0.01 (0.6723 vs 0.6706).
  - The registered rule (raw rows) was followed.
  - A wider lambda grid is a later option, in a fresh registration only.
- **Labels:** the TM rows inherit "posthoc": true from w3_exists_tm.py, which marks test 2's existsT choice. The LR
  rows record a3_lr.py's hash under the key 'w2_lr.py'.
- **Check sentences:** a3_teacher.py's 64 check sentences include one HWU64 TEST text by key ("please turn off the
  lights in the kitchen") and three training texts. They feed only the fp16-vs-fp32 check, never the targets. The
  comment that says "no evaluation text" is wrong.
  - The comment gets fixed, and that sentence replaced, before any later HWU64 test use.
  - The fix waits until 3c's chain has finished, since that chain checks a3_teacher.py's hash.
- **Caveat, kept in every report line:** bge-base's own training data may include these intent sets. potion shares
  this teacher, but the C4 counting table does not, so it could inflate a3B minus ppw_S1000. It does not inflate a3B
  against potion.
