# REGISTRATION T4: a Tsetlin machine slot filler on MASSIVE en-US (written 2026-10-18T08:51:35Z)

Question: can a Tsetlin machine label every word of an utterance with its slot type ("wake me up at [time : five
am]"), reading per-word meaning bits from the static table, and how does it compare with a linear model on the same
bits and with a frozen transformer probe? Novelty (N6): no TM token-level sequence labeler was found; the features
are old in CRF work (cite Haghighi and Klein 2006; Guo et al. 2014; Yu et al. 2013; Saade et al. 2018).

## Test split

MASSIVE 1.1 en-US test, slot labels: 2,974 utterances, 20,287 words. Its SLOT labels have not been read in this line
(its intents were read in T3). They are read once, by the runs below. Everything is fixed here, before that read.

## Fixed (chosen on val, MASSIVE dev, in the exploration rounds E4 to E4d: runs_T4.log, results_slot_tm.jsonl,
## results_slot_lr.jsonl, the study log "Test 4")

- Bits: W+S+P+N+G (19,546 bits per word: word identity, the word's own meaning bits, the previous and next word's
  meaning bits, the sentence's OR bits), from massive_en__slots_K512.npz (massive_slots.py, slot_feats.py).
- Scoring: span F1, the MASSIVE convention (slot_eval.py: convert_to_bio, then exact spans).
- TM: slot_tm.py (sha256 18bce7a2bc1619b2), 8,000 clauses, T 8000, s 8, batch 128, every slot word plus a fresh third of the "Other" words in each epoch
  (--other-keep 0.33), at most 60 epochs, patience 8 evaluations (one every 2 epochs), memory cap 0.2 of the card:
  the setting of E4d_c8000_other33_s1, the best val row of the exploration (val span F1 0.7309, best epoch 52), seeds 1, 2, 3; the epoch with the best val span F1 is restored
  and the test split is scored once at that epoch; --rules 5 (top rules for time, date, place_name, person,
  alarm_type). The exploration runs used the same training code: the 20:44:47Z edit (after round 4 started) only
  added --rules and slot_feats.names(), and the 2026-10-17 23:41:29Z edit only added the saving of the val and
  test predictions.
- LR twin: slot_lr.py (sha256 38b89d92a1d5cb63) on W+S+P+N+G, weight decay chosen on val (0, 1e-5, 1e-4), --test.
- Frozen transformer probe: slot_lr.py minilm3 (frozen all-MiniLM-L6-v2 first-subword vectors of the word and its two
  neighbours, LR), --test.
- Words only: slot_lr.py W (word identity bits only), --test.

Exploration rows (val span F1, seed 1, W+S+P+N+G unless named): LR twin 0.7587; minilm3 0.6751; LR W 0.1915. TM:
batch 512 (W+S+P+N) 0.5678, unstable; batch 128 0.6656; 8,000 clauses 0.7065 (30 epochs); Other 1/3 0.7039 (4,000
clauses, 30 epochs); 8,000 clauses + Other 1/2 0.7223 (60 epochs, best 32); 8,000 clauses + Other 1/3 0.7309 (60
epochs, best 52). The choice is the best val row; the TM's val numbers are one seed, so the test mean may be lower.

## Bars (TM = mean of the 3 seeds; test span F1)

- H1 (beats a frozen transformer probe): TM > minilm3. Strong: every TM seed > minilm3. (Val: 0.7309 (E4d_c8000_other33_s1) against 0.6751.)
- H2 (close to its linear twin): TM >= 0.90 x LR twin. (Val: 0.7309 / 0.7587 = 0.963.)
- Reported, no bar: LR W against LR W+S+P+N+G (what the meaning bits add for a linear model); per-slot F1 of the 10
  most frequent slot types; the readable rules; the published MASSIVE en-US test slot F1, which is not like-for-like
  (models trained on all 51 locales; FitzGerald et al. ACL 2023, Tables 8-9): mT5 81.6, mT5 encoder 80.4, XLM-R 78.7.

## Run

At most 2 jobs of this line at once. Slot 1: t4t_queue.sh lane A (the three LR test rows on CPU, then TM seeds 1, 2, 3).
Slot 2: the test 5 timing pilot (CPU, one thread; no test split), then t4t_queue.sh lane B (TM seeds 3, 2); a seed
runs only once (mkdir claims). Log runs_T4t.log plus one log per run; verdict by verdict_t4.py (sha256 below).
- verdict_t4.py sha256 e52b1df130d60e27 (written before any T4 test read).

## Amendment (wording only, 2026-10-18T08:54:50Z, after lane A started at 08:52:16Z; nothing in the design, bars or code changes)

- "Its SLOT labels have not been read in this line" is imprecise: massive_slots.py (2026-10-17 18:15Z) loaded the val
  and test slot labels once, to build the cache and to check for labels that occur only in val or test (none were
  found; meta "eval_only_labels"). No model was scored on the test slot labels before T4. Read: "not scored before T4".
- Write-up wording (fixed 2026-10-18): H1 = "beats a linear probe on frozen MiniLM word vectors", never
  "beats a transformer", with the fine-tuned published numbers (78.7 to 81.6) next to it. H2 is a 10%
  non-inferiority margin: if it passes, write "X points below LR on the same bits (within the registered 10%
  margin)", never "matches LR".

## Verdict (2026-10-18T17:27:14Z; verdict_t4.py e52b1df130d60e27, output in verdict_t4.txt; slot_tm.py, slot_feats.py, slot_eval.py, slot_lr.py hashes checked: unchanged)

- TM test span F1, seeds 1 / 2 / 3: 69.48 / 70.54 / 69.97, mean 70.00 (val 0.7309 / 0.7298 / 0.7261; best epochs 52 / 50 / 48).
- H1 MET, strong MET: the TM beats a linear probe on frozen MiniLM word vectors (minilm3, 65.95) by 4.05 points, and every
  seed is above it. Fine-tuned published models (FitzGerald et al. 2023, trained on all 51 locales, not like-for-like):
  mT5 81.6, mT5 encoder 80.4, XLM-R 78.7.
- H2 MET: 3.57 points below LR on the same bits (73.57; ratio 0.951, within the registered 10% margin).
- The MASSIVE en-US slot test split is now used for this setting; further tuning on it is post hoc.
