# Step 3c: paraphrase augmentation with the local LLM (plan; not registered yet)

Step 3c of the text-understanding plan. It is registered in the study log after the CPU pilot (section "Tests before
the GPU job") has fixed the prompt and the parser. The GPU job runs at its turn in the queue (decided 22:21Z:
decision_layer's e79, 3a, R3 only if 3a passes, decision_layer's e80, then 3c). The plan was accepted at 22:21Z with one
fix, F1b (section "Filters").

## Question

Do label-preserving paraphrases, written by a local open LLM at training time only, make the TM reader better at
intents on top of the current best stack?

**How to say it, in every report:**
- The LLM (Qwen3.5-9B) and the filter's teacher (bge-base-en-v1.5) are used at training time only. At run time the
  classifier is still a word table, bits and a TM, so it is "network-free at run time". Never write "no neural network
  at any stage" for a 3c model, even on the C4 counting table: its training data contains LLM-written text.
- Any reading of paraphrases is "judged by an AI model, not a person".
- "We found no published study that tests LLM paraphrase augmentation for a Tsetlin machine reader on CLINC150,
  Banking77 or HWU64." Never "first".

## Base stack (R2), fixed now

- **If 3a PASSES:** the base is the a3B table at lambda*. First comes R3's re-tune of the reader on that table: the
  e74 rule at seed 1 over s {2, 4, 8, 12, 16, 24}, then seeds 4-6 at the pick. It is registered as its own short step
  before 3c. 3c then uses that pick and those rows.
- **If 3a FAILS:** the base is ppw_S1000 at 0b's C4 pick (tu/b0_picks.json). The reference TM rows are 0b's seed rows
  at that s.

## Generator

- Qwen3.5-9B Q8_0 (Qwen3.5-9B-Q8_0.gguf, sha256 ff3d8ff1...), Apache-2.0, through llama.cpp b11742's llama-server.
  - It listens on 127.0.0.1 only, with -ngl 99, -np 16 parallel slots, -c 16384 (1,024 tokens per slot), at most 4
    state checkpoints per slot, --jinja and thinking off (--reasoning-budget 0; enable_thinking false per request).
    The client sends 16 requests at a time.
  - The -fa setting is the one the smoke test passed with.
- **It may run only when:**
  - the GPU smoke test has passed: GPU perplexity within 1% of the CPU reference (28.4455), and no CUDA error;
  - the conversion check (llm/hf_ref_ppl.py, registered 21:28:52Z) has said "conversion OK".
- **Input:** the training texts only, read with dl.data.load(name).texts["train"]: CLINC150 15,000, Banking77 9,000,
  HWU64 8,954 (32,954 in all).
  - Banking77's val is carved from the official train split, so its 1,003 val texts are not in "train". They are
    never sent.
  - No val, test or out-of-scope text is ever sent to the LLM.
- **Prompt** (v4, frozen in c3_common.py b858a6fdb8ea9832):
  - System: "You rewrite requests that users send to a virtual assistant. For each request, write 5 different ways a
    user could say it. Keep the same intent and the same details (names, numbers, dates, places). The intent label
    only tells you what the request means: do not add words or details from the label that are not in the request.
    Vary the words and the sentence form; short and casual forms are fine. Do not answer the request. Write one per
    line, with no numbering, no quotes and nothing else."
  - v4 is v3 plus the label sentence. Pilot v3 showed label-word injection: 23.5% of its paraphrases added a word of
    the intent label that the request lacked (Banking77 39%). v4 gives 21.3% (Banking77 34%).
    - v4 removed the worst cases, such as "direct debit" and "after the bank transfer" added to vague requests.
    - It kept others, such as "via the app" and "fiat". Part of what the count catches is natural synonyms ("exchange",
      "card").
    - The rate is reported, not filtered: a filter on label words would also drop the natural synonyms.
  - User: "Intent label: {label}", a new line, "Request: {text}".
  - Why the fixed instructions sit in the system message: for this hybrid model, llama-server keeps state
    checkpoints only at the start of the last user-role message and just before the end of the prompt (read in its
    source). So only text before a user-role message is reused from the prompt cache.
    - Pilot v1 put the label and the text first. Pilot v2 put the instructions first inside a user-role message.
    - Both reused only the 16 tokens of the system message. They were stopped after 5 and 3 requests; their rows are
      kept in tu/c3/pilot_v1/ and tu/c3/pilot_v2/.
  - The label has its underscores replaced by spaces.
  - The label is shown because a known failure of LLM augmentation on these sets is drift to a neighbouring intent
    (Sahu et al. 2022). Labels are training-time knowledge.
- **Decoding:**
  - temperature 0.8, top_p 0.95, top_k 40, min_p 0.05, no repeat penalty, at most 192 new tokens;
  - seed = 1,000,003 x set index + source index + 1, fixed per request (set index 0, 1, 2 for CLINC150, Banking77,
    HWU64).
  - Batched llama.cpp decoding need not be bit-reproducible, so the outputs themselves are the record. They are kept
    with their sha256.
- **Parser:**
  - split the output into lines; strip list markers (digits with "." or ")", "-", "*", "-") and one pair of
    surrounding quotes;
  - drop a line that has no word left after W1's normalisation, a line ending in ":" (a header such as "Here are 5
    ways:"), and a line longer than max(2 x the source's words, the source's words + 12);
  - drop a line equal (by W1's key, B.key) to its source or to an earlier line of the same source;
  - keep the first 5.
- **Output:** tu/c3/gen_{set}.jsonl, one row per source text: index, label, raw output, parsed lines, seconds.
  - Restart-safe: a source already in the file is skipped. The file is appended to, never rewritten.

## Filters (in this order; the counts per set and per filter are reported)

1. **F1, leakage and conflicts.** A paraphrase is dropped when either rule holds:
   - its key equals, or it shares an 8-gram of normalised words with, any text of B.eval_texts() or any MultiWOZ 2.2
     dev or test utterance (read as a3_mwoz_filter.py reads them). B.eval_texts() holds every evaluation split,
     including the three intent sets' val and test, Banking77's carved val, and ABCD and STAR dev and test. These are
     W1's rules, the same as 3a's.
   - its key equals a training original's key: another class is a label conflict, the same class is redundant.
1b. **F1b, near-copies of evaluation texts** (the fix of 2026-10-20 22:21Z).
   - Why: F1's 8-gram rule only sees texts of 8 or more words, and intent utterances are shorter. So a near-copy
     of a val or test sentence that the LLM memorised could pass F1.
   - Rule: a paraphrase is dropped when its bge cosine to some val or test text of its own set is at least 0.98. For
     CLINC150 the out-of-scope val and test texts count too. 0.98 is the threshold of the near-copy measure.
   - It reads those texts only, never their labels, as F1 does. It is conservative: it can only lower a gain.
   - Reported per set: the drops, their share of the F1 survivors, and the natural share of training originals at
     or above 0.98 to the same texts.
2. **F2, the teacher's cosine to its source is at least 0.7** (fixed in the main plan on 2026-10-19).
   - bge-base-en-v1.5 encodes as a3_teacher.py does: [CLS], L2-normalized, fp16 on the GPU, at most 128 tokens.
3. **F3, label consistency** (the registered arm only).
   - A 5-fold cross-fitted multinomial logistic regression (dl.linear, C 10) is fit on the bge-base embeddings of the
     training originals. The folds come from a numpy default_rng(20261021) permutation of the source indices.
   - A paraphrase is kept when the fold model that did not see its source predicts the source's label.
   - Reason: the cosine rule cannot stop drift to a neighbouring intent, which stays close in cosine. Filtered LLM data
     helped Banking77 more than unfiltered data (+1.1 against +0.4 with BERT-base, Findings of EMNLP 2025).

## Arms

- **3c-F (registered):** originals plus the paraphrases kept by F1, F1b, F2 and F3.
- **3c-C (descriptive, LR only, cheap):** originals plus the paraphrases kept by F1, F1b and F2 (the main plan's
  rule plus the leakage fix). It shows what the label filter does.
- **Reference:** originals only, on the same base.
- **Count-only twin (descriptive; runs only if 3c-F passes, after the verdict):**
  - 5 noisy copies of each original: random deletion of each word with p 0.1, plus one random swap of two adjacent
    words (EDA's two lexicon-free operations; numpy default_rng(20261022)), F1 only;
  - TM and LR. It says how much of a gain needs the LLM's meaning.

## Features and readers

- **c3_feats.py:** the base table's feature code (w1/wordtab_feats.py's path, unchanged) is run again for the original
  splits.
  - The k-means vocabulary stays the ORIGINAL training tokens, so the 512 concepts are the reference's.
  - The kept paraphrases are pooled with the same table and centroids, and appended to "train" with their source's
    label. n_orig is recorded.
  - **Equivalence check:** the rebuilt original features must equal the cached reference features bit for bit (the
    largest absolute difference is reported).
    - If they do, the base's existing rows are the reference.
    - If not, the reference TM and LR rows are run again on the rebuilt features (the same s, seeds 4-6), and those
      rows are the reference.
- **c3_exists_tm.py:** b0_exists_tm.py with two changes and nothing else.
  1. The existsT thermometer is fit on the n_orig original training rows only, so the bit definition stays the
     reference's. More examples are the only change.
  2. The output files and the study label ("3c").
  - The TM settings are unchanged: 4,000 clauses, T 4000, 40 epochs, batch 128, val at the best epoch.
  - It runs at the base's s, on seeds 4-6.
- **c3_lr.py:** w3_lr.py with the same thermometer change (C in 1/10/100, best on val).

## Bar (fixed now)

- **PASS:** 3c-F minus the reference is at least +0.5 with BOTH probes on at least 2 of the 3 sets.
  - TM: the mean of the paired gains over seeds 4-6. LR: val.
  - Unrounded gains are compared with 0.5 - 1e-9.
- **Otherwise FAIL.**
- **Descriptive:**
  - 3c-C's LR gains;
  - the keep rates per filter;
  - 30 kept and 30 dropped paraphrases per set (judged by an AI model, not a person);
  - the near-copy rates: before F1b (its drops among the F1 survivors, against val and test), after F1b (c3_nearcopy.py
    on the kept paraphrases against val: 0 at 0.98 by construction, with the 20 closest pairs), and the original
    training texts' natural rate;
  - the per-seed TM gains;
  - the share of the gap to potion that is closed;
  - the training seconds.
- **On PASS:**
  - R3: the reader is re-tuned on the augmented stack.
  - R4: the agent is a separate frozen plan in the decision_layer study, because paraphrasing dialogue turns is another job.
  - R5: the guards.

## Tests before the GPU job

- **Pilot results** (study log 22:32:39Z and the registration; 20 training texts per set, CPU, no probe):

  | | v3 | v4 (frozen) |
  |---|---|---|
  | answers with 5 paraphrases | 58 of 60 | 56 of 60 |
  | kept by F1-F3 (3c-F) | 257 of 298 (86%) | 248 of 296 (84%) |
  | dropped: equal to an evaluation text (F1) | 7 | 10 (+1 by 8-gram) |
  | dropped: equal to a training original (F1) | 12 | 15 |
  | dropped: near-copy (F1b) | 2 | 4 |
  | dropped: cosine < 0.7 (F2) | 7 | 5 |
  | dropped: another label (F3) | 13 | 13 |
  | label-word injection | 23.5% | 21.3% |

  - Natural near-copy share of the training originals against the same val and test texts: CLINC150 3.75%,
    Banking77 8.78%, HWU64 1.66%.
  - All drops of v3 were read (judged by an AI model, not a person). F1's drops are common phrasings found verbatim in
    val or test. F3 mostly catches real drift, with a few false drops.
- **CPU pilot** (training texts only, no probe):
  - a CPU llama-server (-ngl 0, 2 threads, nice 19, behind the CPU gate) on 20 training texts per set, 60 requests;
  - measures the parse rate and the F1 and F2 keep rates (bge on CPU);
  - 30 outputs are read (judged by an AI model, not a person).
  - The pilot may change the prompt and the parser. Every change is logged before the registration. The pilot's rows
    are not used for training.
- **Code checks:**
  - the parser on fixed strings;
  - F1 on planted duplicates;
  - F3's folds: no source is judged by a model that saw it.
- **c3_exists_tm.py and c3_lr.py:** with no paraphrases, their rows must equal b0_exists_tm.py's and w3_lr.py's on
  the same features and seed. The TM part of this runs inside the GPU job, as its first run.

## Budget (one big GPU job at a time; CPU gate cpu_load.py at most 4.5 cores)

1. Generation: 32,954 requests, about 1.5M new tokens (the pilot's answers were 38-49 tokens).
   - At an assumed 400-700 tokens/s over 16 slots this is about 0.6-1 GPU-hour. It is measured on the first 1,000
     requests.
   - If the projected total is over 4 hours, the job stops and the plan is revised.
2. Filters and features: about 15 minutes.
3. 9 TM runs on 2-6 times the data: about 15-40 minutes. 0b's seed-1 runs at s 8 took 42 s (CLINC150) and 25 s
   (Banking77) of training on the originals; s 2 is slower (157 s).
4. LR fits: about 10 minutes.

In all, about 1.5-2.5 GPU-hours, most of it generation.

## Prior

About 30% (plan). The training sets are full size, and the literature gives +0.3 to +1.5 with full data.

## Newest work (searched 2026-10-20; adds to the 2026-10-19 search in novelty/text_understanding_prior_art.md)

- **DDAIR** (Castillo-López, Lombard, Semmar, de Chalendar; EACL 2026; arXiv 2601.11234, 2026-01-16).
  - Class-guided LLM augmentation (Mistral-7B-Instruct, Llama-3-8B-Instruct) for intents in 2-shot and 5-shot
    settings, on BANKING77, CLINC150 and MPGT, with a BERT-base classifier.
  - An example is "ambiguous" when its nearest class centroid (bge-base-en-v1.5, MPNet or MiniLM) is another intent.
  - Regenerating ambiguous examples beat unfiltered augmentation by +0.2 to +1.5 macro-F1 on BANKING77 and CLINC150.
    For example, BANKING77 2-shot with Mistral went from 56.62 to 57.43.
  - DROPPING them was worse on those two sets (51.12 on the same cell). There the centroids come from 2-5 examples.
  - Our regime is full data (CLINC150 100 originals per class; Banking77 31-168, median 114; HWU64 33-159, median
    156), so F3 is fit on far more data. It is still the
    reason 3c-C (no label filter) is reported beside 3c-F.
- **Sahu et al. 2022** (NLP4ConvAI, GPT-3, few-shot): augmentation helps less when intents are close, because the
  generator writes a neighbouring intent.
- **PromDA** (ACL 2022): consistency filtering by relabelling with an NLU model.
- **Not found:** a 2026 study of LLM augmentation for a TM, bag-of-words or static-table intent classifier; or one on
  the full CLINC150, Banking77 or HWU64 sets with a static student. This agrees with the 2026-10-19 search.
- **Current state of the art on the three sets** (full data, test; from the 2026-10-19 search): CLINC150 97.80,
  Banking77 94.77, HWU64 94.33 (SPACE-2, 2022). Our numbers are val, so this is placement only.
