# Plan e71: local guideline proofs around real conversations, in word edits

A new study, after e70. Dev and proof work only; no test split is read. Frozen (`scripts/freeze.py`) before any run, with the code's hashes. Changes after the freeze go into a dated note.

**Publication note:** the wording of this copy was edited for publication on 2026-10-23. The registered question, rules, thresholds, seeds, analysis plans, amendments, dates and times are unchanged; the freeze record next to this file holds the SHA-256 of the text as registered.

## Why

e70 showed that over the Level-B superset, with the text bits free, no guideline rule holds for the raw TM. The solver builds texts that look nothing like real conversations: their nearest dev row is about 1,500 text bits away. The local question is the meaningful one. A Hamming ball in bits is not a good threat model for text: one token sets a median of 67 bits in each word-table field (`logs/e71_word_units.out`), so a ball of 20 bits is smaller than one word. So the main measure is word edits, encoded exactly:

> Around a real conversation state, how few word edits to the customer's latest message make the agent choose a button that the rules forbid in that state?

Prior art: Przybysz et al. 2023 verify local (Hamming) robustness of vanilla TMs by SAT; this is cited. What is new here is guideline compliance of a dialogue policy under word edits, measured around real states (search scope in the novelty report in the project notes, not published).

## The edits (fixed now)

- **Where:** the customer's latest utterance u, the last " | " part of the state's customer text. It is also the last customer segment of the last 3 turns in 3,616 of 3,684 dev states. Elsewhere only the customer field changes, and the words and last-3-turns bits stay as logged.
- **An edit** is one of:
  - deleting one of u's words (a `\w+` token, as the word lexicon reads them);
  - inserting one candidate word at the end of u.
- u keeps at least one word, since an empty text would read as "(nothing yet)".
- **Candidate words:** every word found in at least 5 train customer utterances, 1,626 words.
- **The memory bits** are fixed as logged; the history is real.
- **k list:** 1, 2, 3, 5 edits.

## The text model (`scripts/verify_local.py`, TextModel)

- **xcust and xrecent** (lookup + OR): the bits of the fixed text, OR the codes of the kept words, OR the codes of the inserted words. Exact.
- **Word unigrams:** present in the fixed text, or a kept or inserted word. Exact.
- **Word bigrams:**
  - those inside the fixed text stay on;
  - another lexicon bigram (a, b) may be on only if a is in the region (the kept and inserted words, or the fixed word just before u) and so is b (or the fixed word just after u);
  - with no edit, every bigram equals the real state's.
  - This is a superset of the real outcomes, because the insertion order is left free.
- **Pre-check, passed** (`results/explore/verify_e71.jsonl`, tag e71_model_smoke, 01:29 UTC): on 200 random dev states the model's text bits equal `view.transform` exactly.
  - with no edit: 200 of 200;
  - with a random edit set (about 30% of u's words deleted, 0 to 3 candidate words appended, the order known): 200 of 200.

## The MILP, one per (state, rule group)

- **Variables:** keep (one per word of u), insert (one per candidate), the text bits that can change, the clauses (e70's exact TM encoding, with the fixed bits evaluated away), and one selector per button of the group.
- **Objective:** the number of edits.
- **Constraint:** some button of the group wins.
  - With big-M rows per pair, M = an exact bound from the clause weights.
  - P_first: B beats the 5 allowed buttons. P_prec: B beats all 29 others. The tie rule is e70's.
- **Rule groups per state** (e70's rules, `scripts/lineb_counts.properties`):
  - with an empty history, P_first (25 forbidden buttons);
  - otherwise one group per missing prerequisite A (pull-up-account, enter-details, verify-identity, log-out-in, search-faq), holding the buttons with a rule (A, B).
- **Outcomes:**
  - **certified for every edit set** (infeasible): no deletion of u's words and no insertion of candidate words makes any button of the group win;
  - **edits found** (optimal k*): certified for every k below k*;
  - **bounds only** (time limit): the dual bound gives a certified k (no edit set below it), and the best edit set found is reported.
- **Replay:** every edit set found is applied to the real text (deleted words removed, inserted words appended), then `view.transform`, then the model. It is called realizable only if the model then chooses a button of the group.
- **Skipped:** a state where the model's own choice already breaks a rule of the group is counted, not solved.

## Pre-check of the MILP (CPU phase 2, then the freeze with every script hash, then the runs)

On 30 random dev states with a random real edit set fixed:
- the model's own choice on the edited text must be feasible: the encoding contains the real outcome;
- the count of runner-up choices that come out infeasible is reported. The superset on bigrams may let some be feasible.
- If the first check fails anywhere, the code is fixed and the pre-check rerun before the freeze.

**Pre-check results (before the freeze):**
- **"Contains real", with no solver** (`--check contains`): the MILP's rows are evaluated at the real outcome of a random real edit set.
  - First run (01:34 UTC, 60 states): 54 ok, 6 broken. Every broken row was a bigram "no edit" row (|x - x0| <= E), written with the edit terms' signs reversed.
  - Fixed in two steps: 191 of 200 after the first fix, then 200 of 200 with fixed bits matching 200 of 200 (01:49, tag e71_contains_precheck3).
  - The earlier rows stay in `results/explore/verify_e71.jsonl` (tags e71_contains_precheck, e71_contains_debug, e71_contains_precheck2).
- **With HiGHS** (`--check milp`, 02:06 to 02:09 UTC, 30 states, a random real edit set fixed, tag e71_milp_precheck):
  - the model's own choice on the edited text is feasible in 30 of 30;
  - the runner-up is infeasible in 28 of 30, and the bigram superset lets 2 be feasible;
  - at most 3.3 s per MILP; median 239,466 rows and 2,765,108 nonzeros.
- **The text model** (tag e71_model_smoke): exact on 200 of 200 states, with no edit and with random edits.

## States

- **The sample:** 200 dev states, fixed by `numpy.random.default_rng(0)` (`sample_states`): 60 of the 1,004 first-action states and 140 of the 2,680 later ones.
- **Groups:** 60 P_first; 519 P_prec groups (134 enter-details, 136 log-out-in, 112 search-faq, 103 verify-identity, 34 pull-up-account).
- So 579 MILPs per model.

## Models, order, budget (fixed now)

- **Order:**
  1. TM seed 1;
  2. TM seeds 2 to 5;
  3. secondary: the bit-flip minimal distance for TM seed 1 on the same states (e70's encoding plus a Hamming objective over the text bits, memory fixed), reported with the calibration (one token is about 67 bits per field).
  - Then **e71b** (fixed now, so it cannot be shaped later):
    - MLP-128 only (e70's twin), on the same 200 states, with the same edit model, the same 120 s cap and the same outcome categories, run after the TM part;
    - time-outs are reported as "unknown", with their dual bounds;
    - the MLP text-model encoding and its pre-check go into e71b's own dated note, before its first run.
- **Time limit:** 120 s per MILP; 24 h wall clock for e71.
- **CPU:** phase 2 began at 02:04 UTC, when the phase-1 rule was lifted early (2.21 cores, 0 of 60 periods throttled).
  - At most 2 solver processes with 2 threads each, over e70 and e71 together, while the total stays under about 5 cores and unthrottled.
  - The load is measured before each start (`scripts/cpu_load.py`: 3.43 cores, 0 of 100 throttled at 02:05:20 UTC). Above 5 cores, back to one solver process.
- Whatever is not run is reported as not run.

## Reported

- **Per model:** for k = 1, 2, 3, 5, the share of the 200 states certified (every group of the state certified at k), and the same per group.
- **Distributions:**
  - k* per (state, group) where found;
  - the dual bounds where only bounds exist;
  - the count certified for every edit set;
  - the skipped states;
  - solve times (median, p90, max) and the time-limit count.
- **Realizability:** the replayed counterexamples, with their actual word edits as examples.
- **States whose latest customer utterance is outside the last 3 turns** (only the customer field changes; 68 of 3,684 on dev): reported separately, with their own counts.
- **The early timing sign from e70:** the first MLP-128 check was still running after 3 minutes, where the TM's took 0.6 s. It is reported as measured, whatever it ends as.

## Wording

- **The headline, exactly:** "for X% of 200 sampled real dev states, no deletion of words from the customer's latest message, or insertion of up to k of 1,626 common customer words at its end, makes the agent choose a button the rules forbid in that state".
- Name the rules, the edit model (where, which words) and the sample.
- Never write "any rewording", "any customer", "robust to any rewording", "for all texts", or any "first".
- Cite Przybysz et al. 2023 next to any statement on TM robustness verification.
