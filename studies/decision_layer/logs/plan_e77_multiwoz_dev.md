# Dev plan e77 (2026-10-19): MultiWOZ 2.2 dev, both agents tuned the same way, before R24 (rules fixed before the first GPU job)

Scope: `logs/plan_e77_third_dataset_scope.md`, fixed at 21:14 UTC with its additions, and the counts.
This plan is frozen with `scripts/freeze.py` before the first GPU job, with the contamination result cited in it. Dev
only: no test example exists yet, and no test label is read.

**Publication note:** the wording of this copy was edited for publication on 2026-10-23. The registered question, rules, thresholds, seeds, analysis plans, amendments, dates and times are unchanged; the freeze record next to this file holds the SHA-256 of the text as registered.

## Data (as prepared, 21:19 UTC)

- 128 labels: the sorted domain-act set per system turn, slots dropped, combinations under 20 train occurrences ->
  OTHER.
- Train: 8,437 dialogues, 56,581 examples. Dev: 1,000 dialogues, 7,359 examples.
- OTHER is 3.0% (train) and 3.4% (dev). Majority-label accuracy on dev: 7.16%.
- data/multiwoz/examples_{train,dev}.json.gz (0e20a314, 3414cc8a).
- **The contamination check** (from the text-understanding line; project notes, not published: script mwoz_c4_overlap.py 1d5ab407, result
  overlap.json 52cb74eb, 21:42 UTC; checked independently at 21:43 UTC). W1's dedupe rule (a verbatim sentence key
  or a shared 8-gram) was run on all 2,243,398 C4 S1000 documents behind the table, against MultiWOZ 2.2 dev and test
  utterances, read from data/untrusted/multiwoz22 (the 4 file hashes in overlap.json equal those of the study's MANIFEST).
  - Dev: 365 of 14,748 utterances (2.47%) match a C4 sentence: 140 verbatim and 228 by an 8-gram.
  - Test: 385 of 14,744 (2.61%): 159 verbatim and 232 by an 8-gram.
  - Whole utterances of 8 or more words equal to a C4 sentence: 3 of 11,723 in dev and 6 of 11,715 in test.
  - The matches are stock service phrases ("Thank you, good bye!", "is there anything else I can help you with"),
    not MultiWOZ dialogues found in C4.
  - **Reading:** a small, declared overlap of common phrases. The table is not rebuilt. This is cited again in R24.

## Done before the freeze (CPU only; no model trained, no score computed)

- **`--dataset` choices** (21:26:08 UTC): abcd_explore.py and abcd_confirm.py got "multiwoz" in their `--dataset`
  choices, under the pre_R20 protocol.
  - Pristine copies: logs/pre_e77/abcd_explore.py (d3c53f57) and logs/pre_e77/abcd_confirm.py (0ae66772).
  - Diffs: logs/pre_e77/diff_abcd_explore.txt (236ea6ff) and diff_abcd_confirm.txt (26e0f832). Each is -1/+1, the
    choices line only; its help text names the new choice. New hashes: abcd_explore.py ca6b017c, abcd_confirm.py
    8c86665a. For abcd and star nothing else can change.
  - dl.abcd.load_examples already reads data/<dataset>/examples_<split>.json.gz, so no other file changes.
  - Neither file is imported by the running e71 seed 3 or by e76 (verify_enum.py, verify_enum_c4.py,
    verify_local.py, verify_policy.py and the modules they import). The e71 v2 and e76 freeze records list both files
    with their old hashes; this note is the reason they differ.
  - The checkers of R21-R23 hashed the old files. Those checks have run, and this change is recorded.
- **`data/emb/multiwoz__none.npz`** (fe366dc1) from none_prepare.py. No text is read; it is byte-identical to
  abcd__none.npz.
- **Coverage** (`scripts/e77_coverage.py`, count_table_coverage.py's count; logs/e77_coverage.out 98c11f30,
  results/explore/count_table_coverage_multiwoz.json de72aad2). Tokens with no vector in the C4 table:
  - cust: 0.35% (train) and 0.40% (dev);
  - recent: 1.10% (train and dev);
  - mostly Cambridge place names (broxbourne, stortford, gonville, addenbrookes), "gastropub" and the phone prefix
    01223;
  - no text loses all its tokens;
  - the tokenizer port matches the reference on 127,880 of 127,880 texts.
  - ABCD and STAR were 0.46-0.87%.
- **View smoke test** (`scripts/e77_view_smoke.py`; logs/e77_view_smoke.out 9c1f2f4d and
  e77_view_smoke_2threads.out): the C4 agent's view B builds on MultiWOZ as under table_run.py.
  - 6,580 bits: mem 262, words 1,000, xcust 2,673, xrecent 2,645 (STAR's view B has 7,227, ABCD's 6,684).
  - Bits SHA-256: train a8c92a30, dev 311e8933; equal with 1 and 2 threads.
  - e77_pick.py stops if a C4 row has other bits, or if the bge-large rows do not all share one set of bits.

## Code and cache for the GPU (after the freeze)

- **`data/emb/multiwoz__bge-large.npz`** from `scripts/multiwoz_embed.py` (a1702d7b). It is star_prepare.py's
  embedding loop: bge-large-en-v1.5, normalised; cust, recent and full, with "(nothing yet)" for an empty text.
  - It runs on the GPU, about 20 to 30 min, train and dev only, as the first job of e77a.
  - At R24, `--add-test` embeds only the test texts. It copies the train and dev arrays from the dev-stage file,
    which is kept as data/emb/multiwoz__bge-large.devstage.npz, and checks them byte for byte. So the agents tuned on
    dev see the same vectors at R24.

## The screens (MultiWOZ dev, 4,000 clauses, T 4,000, seed 1, R21's other settings)

- **The C4 agent:** view B on the C4 table through table_run.py, `--encoder none`, at s 8, 12, 16 and 24.
- **The bge-large agent:** R18's view (`--encoder bge-large --parts mem,intent,step,words`) at s 8, 12, 16 and 24.
- **LR** on view B's bits, on R21's wide C grid (abcd_confirm.py `--dry-run`: C chosen on dev and scored on dev).
  This has no rule.
- **Jobs:** `logs/jobs_e77a.txt` (a8b728e7): 10 jobs, one at a time, own log `logs/e77a_queue.log`, cpu_load gate
  4.5. The first is the bge-large cache; then the 8 screens; then LR. Every TM model is saved (`--save`, results/explore/abcd_models/), for the analyses of where the text part
  fails (dev, no rule). This was added after an independent check (21:34 UTC); the earlier file is in logs/archive/.
- **The pick** (`scripts/e77_pick.py`, 08a1cb6c, `--stage screen`), for each agent:
  - the highest 4k dev score, if it is at least 0.3 points above the agent's own s 8;
  - otherwise s 8;
  - an exact tie goes to the first of [8, 12, 16, 24].
  - It was dry-tested on made-up rows in a temporary folder: the 0.3 margin, the tie, the job file, the refusal to
    overwrite, the freeze time and the bits checks.

## The 16k step (MultiWOZ dev, 16,000 clauses, T 16,000, seed 1)

- **Runs:** for each agent, s 8, and its pick if that is not s 8. That is up to 4 jobs (`logs/jobs_e77b.txt`, written
  by e77_pick.py).
- **s\*** for each agent (`e77_pick.py --stage final`) is its pick, unless its 16k dev is below its own s 8 at 16k;
  then s\* = 8.
- **Reported, with no rule:**
  - the tuned dev gap at 16k, C4 at its s* minus bge-large at its s*;
  - both agents against the majority label (7.16%);
  - the LR dev number;
  - where the text part fails (added at 21:34 UTC): `scripts/e77_breakdown.py` (00aa17dc) on both
    agents' s* models at 16k. It covers first against later actions, per label, the main confusions, and "domain
    right" against "act right". Its bits and CPU accuracy are checked against each row first. It was tried on STAR
    dev (e74c against e74d; both checks exact).

## Then R24 (written and frozen separately; the scope's rules)

- **Test use:** MultiWOZ's test split, 1st use.
- **Arms:** both agents at their s*, 16k, seeds 1 to 5.
- **R24.1, superiority:** C4 minus bge-large above 0, with the whole 95% interval above 0.
- **R24.2:** at least -2.0.
- **No rule:** each agent against its own s 8, and TM against LR.

## Budget and order (after the N9 study tonight)

- **GPU:** the view sizes and train size are close to STAR's, so STAR's times are the guide:
  - the bge-large cache, about 25 min;
  - 4 C4 screens at about 5 to 6 min, and 4 bge-large screens at about 2 min;
  - up to 4 16k runs: C4 about 20 min, bge-large about 8 min;
  - about 2 h in all.
- **CPU:** minutes.
- **Order:** this plan frozen (with the contamination result) -> the bge-large cache -> e77a -> pick -> e77b ->
  s* -> R24 frozen -> R24 runs.

## Selection caveat

s is chosen on MultiWOZ dev among 4 values for each agent, by the same rule. The 16k step can only fall back to s 8.
